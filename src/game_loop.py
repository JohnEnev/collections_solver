"""
Core game loop for playing NYT Connections puzzles.
"""

import torch
from typing import Dict, List, Any, Optional, Tuple
from .utils import extract_json_from_response, extract_guess_from_parsed, check_one_away


def build_prompt(
    remaining_words: set,
    found_groups: Dict[str, List[str]],
    relevant_wrong: List[Dict],
    one_away_guesses: List[Dict]
) -> str:
    """Build the prompt for the model based on current game state."""
    
    context = ""
    if found_groups:
        context += "FOUND: " + ", ".join([f"{n}: {w}" for n, w in found_groups.items()]) + "\n\n"
    if relevant_wrong:
        wrong_str = ", ".join([str(g['words']) for g in relevant_wrong])
        context += f"WRONG (do NOT guess these again): {wrong_str}\n\n"
    
    remaining_list = ", ".join(sorted(remaining_words))
    
    if one_away_guesses:
        # ONE AWAY recovery prompt
        last_one_away = one_away_guesses[-1]['words']
        already_tried = [g['words'] for g in relevant_wrong]
        other_words = sorted(remaining_words - set(last_one_away))
        
        prompt = f"""{context}REMAINING: {remaining_list}

Your guess {last_one_away} was ONE AWAY — exactly 3 words are correct, 1 is wrong.
Already tried: {already_tried}

Which word is the impostor? Consider:
- Which word has the weakest connection to the theme?
- Which word might belong to a different group?

Pick ONE word to swap out, replace it with one of: {other_words}
Output ONLY valid JSON: {{"group": ["W1", "W2", "W3", "W4"]}}"""
    else:
        # Standard prompt
        prompt = f"""{context}REMAINING: {remaining_list}

Find 4 words that share a hidden theme.

Think step by step:
1. What patterns do you see? (categories, wordplay, phrases, etc.)
2. Which 4-word group are you MOST confident about? Publish this one.
3. Verify: Are all 4 words in the REMAINING list above?

Output your most confident group ONLY in valid JSON:
{{"group": ["W1", "W2", "W3", "W4"]}}"""
    
    return prompt


def play_game(
    puzzle: Dict[str, Any],
    model,
    tokenizer,
    max_mistakes: int = 4,
    base_temp: float = 0.3,
    max_retries: int = 12,
    return_trace: bool = False
) -> Dict[str, Any]:
    """
    Play a single NYT Connections game.
    
    Args:
        puzzle: Dict with 'words' and 'solution' keys
        model: The language model
        tokenizer: The tokenizer
        max_mistakes: Maximum wrong guesses allowed (default 4)
        base_temp: Base temperature for generation (default 0.3)
        max_retries: Maximum total attempts including duplicates (default 12)
        return_trace: Whether to return full game trace for analysis
    
    Returns:
        Dict with 'solved', 'groups_found', 'mistakes', and optionally 'trace'
    """
    remaining_words = set(w.upper() for w in puzzle["words"])
    found_groups = {}
    mistakes = 0
    previous_guesses = []
    tried_combinations = set()
    total_attempts = 0
    consecutive_duplicates = 0
    trace = [] if return_trace else None
    valid_outputs = 0
    invalid_outputs = 0
    
    # Build solution lookup
    solution_groups = {
        frozenset(w.upper() for w in members): name 
        for name, members in puzzle["solution"].items()
    }
    
    while len(found_groups) < 4 and mistakes < max_mistakes and total_attempts < max_retries:
        total_attempts += 1
        
        # Auto-complete when 3 groups found
        if len(found_groups) == 3:
            final_set = frozenset(remaining_words)
            if final_set in solution_groups:
                found_groups[solution_groups[final_set]] = list(remaining_words)
                if trace is not None:
                    trace.append({
                        "turn": total_attempts,
                        "action": "AUTO_COMPLETE",
                        "group": list(remaining_words)
                    })
                break
        
        # Temperature: hail mary on last mistake, or bump after duplicates
        if mistakes == max_mistakes - 1:
            temperature = 0.7
        else:
            temperature = min(0.9, base_temp + (0.1 * consecutive_duplicates))
        
        # Filter relevant wrong guesses (still in play)
        relevant_wrong = [
            g for g in previous_guesses 
            if set(g['words']).issubset(remaining_words) and 'DUPLICATE' not in g['feedback']
        ]
        one_away_guesses = [g for g in relevant_wrong if "ONE AWAY" in g['feedback']]
        
        # Build prompt
        prompt = build_prompt(remaining_words, found_groups, relevant_wrong, one_away_guesses)
        
        # Generate response
        messages = [
            {"role": "system", "content": "NYT Connections. Find groups of 4. Output JSON."},
            {"role": "user", "content": prompt}
        ]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2000,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse response
        parsed = extract_json_from_response(response)
        
        turn_record = {
            "turn": total_attempts,
            "prompt": prompt,
            "response": response,
            "parsed": parsed,
            "temperature": temperature
        } if trace is not None else None
        
        if not parsed:
            mistakes += 1
            invalid_outputs += 1
            consecutive_duplicates = 0
            if turn_record:
                turn_record["result"] = "PARSE_FAIL"
                trace.append(turn_record)
            continue
        
        guess = extract_guess_from_parsed(parsed)
        if not guess or len(guess) != 4:
            mistakes += 1
            invalid_outputs += 1
            consecutive_duplicates = 0
            if turn_record:
                turn_record["result"] = "INVALID_GUESS"
                trace.append(turn_record)
            continue
        
        valid_outputs += 1
        guess_list = [w.upper() for w in guess]
        guess_set = frozenset(guess_list)
        
        if turn_record:
            turn_record["guess"] = guess_list
        
        # Check for duplicate
        if guess_set in tried_combinations:
            consecutive_duplicates += 1
            if turn_record:
                turn_record["result"] = "DUPLICATE"
                trace.append(turn_record)
            previous_guesses.append({"words": guess_list, "feedback": "DUPLICATE"})
            continue
        
        consecutive_duplicates = 0
        tried_combinations.add(guess_set)
        
        # Check if words are valid
        if not guess_set.issubset(remaining_words):
            mistakes += 1
            if turn_record:
                turn_record["result"] = "INVALID_WORDS"
                trace.append(turn_record)
            previous_guesses.append({"words": guess_list, "feedback": "INVALID"})
            continue
        
        # Check if correct
        if guess_set in solution_groups:
            group_name = solution_groups[guess_set]
            found_groups[group_name] = guess_list
            remaining_words -= guess_set
            if turn_record:
                turn_record["result"] = "CORRECT"
                turn_record["group_name"] = group_name
                trace.append(turn_record)
        else:
            is_one_away = check_one_away(guess_set, solution_groups)
            mistakes += 1
            feedback = "ONE AWAY" if is_one_away else "WRONG"
            if turn_record:
                turn_record["result"] = "ONE_AWAY" if is_one_away else "WRONG"
                trace.append(turn_record)
            previous_guesses.append({"words": guess_list, "feedback": feedback})
    
    result = {
        "solved": len(found_groups) == 4,
        "groups_found": len(found_groups),
        "mistakes": mistakes,
        "attempts": total_attempts,
        "valid_outputs": valid_outputs,
        "invalid_outputs": invalid_outputs,
    }
    
    if return_trace:
        result["trace"] = trace
    
    return result

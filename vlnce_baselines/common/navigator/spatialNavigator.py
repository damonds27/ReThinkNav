import re
import random
from vlnce_baselines.common.navigator.api import *
from vlnce_baselines.common.navigator.prompts import *
from vlnce_baselines.common.navigator.clip import *

class ReThinkNav():
    def __init__(self, device, llm_type, api_key):
        self.device = device
        self.llm = llmClient(llm_type, api_key)
        self.spatial = spatialClient(self.device)
        
    # =====================================
    # ===== Instruction Comprehension =====
    # =====================================

    def get_actions(self, instruction):
        response = self.llm.gpt_infer(ACTION_DETECTION['system'], ACTION_DETECTION['user'].format(instruction))
        return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip() if '<think>' in response else response

    def get_landmarks(self, actions):
        actions = actions.replace("\n", " ")
        response = self.llm.gpt_infer(LANDMARK_DETECTION['system'], LANDMARK_DETECTION['user'].format(actions))
        return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip() if '<think>' in response else response
    
    # =============================
    # ===== Visual Perception =====
    # =============================
    def observe_environment(self, logger, current_step, images_list):        
        observe_results = []
        observe_dict = {}
        for direction_idx, direction_image in images_list.items(): 
            observe_result = self.spatial.observe_view(logger, current_step, direction_idx, direction_image)
            print("observe_result:", observe_result)
            logger.info(observe_result)
            observe_results.append(observe_result) 
            observe_dict[direction_idx] = observe_result
        return observe_results, observe_dict
    
    # ===================================
    # ===== Progress Estimation =========
    # ===================================


    def save_history(self, logger, current_step, next_vp, thought, curr_observe, nav_history):
        # ===== get observation summary =====
        direction_id = int(curr_observe.split("Direction Viewpoint")[0].replace("Direction","").strip())
        direction = DIRECTIONS[direction_id]
        curr_observe = "Scene Description"+curr_observe.split("Scene Description")[1]
        print("curr_observe: ", curr_observe)
        obs_response = self.llm.gpt_infer(OBSERVATION_SUMMARY['system'], OBSERVATION_SUMMARY['user'].format(curr_observe))
        observation = f"Direction {direction} " + (re.sub(r'<think>.*?</think>', '', obs_response, flags=re.DOTALL).strip() if '<think>' in obs_response else obs_response)
        
        # ===== get thought summary =====
        thought_response = self.llm.gpt_infer(THOUGHT_SUMMARY['system'], THOUGHT_SUMMARY['user'].format(thought))
        thought = re.sub(r'<think>.*?</think>', '', thought_response, flags=re.DOTALL).strip() if '<think>' in thought_response else thought_response
        
        # ===== get nav history =====
        nav_history.append({
            "step": current_step,
            "viewpoint": next_vp,
            "observation": observation,
            "thought": thought
        })
        logger.info(f"The history at current step is {nav_history}")
        return nav_history
    
    def review_history(self, logger, nav_history):
        nav_history_str = " -> ".join(["Step "+str(idx)+" Observation: "+item["observation"]+" Thought: "+item["thought"] for idx, item in enumerate(nav_history)])
        logger.info("History: " + nav_history_str)
        return nav_history_str

    def estimate_completion(self, logger, actions, landmarks, history_traj, observation, last_estimation, movements):
        # print(COMPLETION_ESTIMATION['user'].format(history_traj, landmarks, observation, actions, last_estimation, movements))
        response = self.llm.gpt_infer(COMPLETION_ESTIMATION['system'], COMPLETION_ESTIMATION['user'].format(history_traj, landmarks, observation, actions, last_estimation, movements))

        clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip() if '<think>' in response else response
        if "Executed Actions" in clean_response:
            logger.info("Executed Actions " + clean_response)
            if "Executed Actions:" in clean_response:
                return clean_response.split("Executed Actions:")[1].strip()
            else:
                return clean_response.split("Executed Actions")[1].strip()
        else:
            return clean_response

    
    # ===================================================
    # ===== Decision-making under No Loop Condition =====
    # ===================================================
  

    def move_to_next_vp(self, logger, current_step, instruction, actions, landmarks, history_traj, estimation, observation, observe_dict):    
        break_flag = True
        score_text = ""  
        pred_vp = None
        valid_viewpoints = set(map(str, observe_dict.keys()))
        for i in range(2):
            effective_prediction, thought_list = [], []
            batch_responses = self.llm.gpt_infer(NAVIGATOR['system'], 
                                                  NAVIGATOR['user'].format(observe_dict.keys(), current_step, instruction,
                                                                           actions, landmarks, history_traj, estimation, observation),
                                                  num_output=3)
            for decision_reasoning in batch_responses:
                clean_reasoning = re.sub(r'<think>.*?</think>', '', decision_reasoning, flags=re.DOTALL).strip()
                if "Prediction:" not in clean_reasoning:
                    continue
                logger.info(f"================retry id {i} in pred_vp==========")
                logger.info(clean_reasoning)
                
                try:
                    parts = clean_reasoning.split("Prediction:")
                    if len(parts) < 2:
                        raise ValueError("No Prediction found in LLM output.")

                    thought_and_score_text = parts[0].strip()
                    raw_pred_vp = parts[1].strip()
                    
                    if "Score:" in thought_and_score_text:
                        thought_text, score_text = thought_and_score_text.split("Score:", 1)
                        pred_thought = thought_text.strip()
                        score_text = score_text.strip()
                    else:
                        pred_thought = thought_and_score_text
                        score_text = ""
                        
                    pred_vp = re.sub(r'[^0-9]', '', raw_pred_vp)
                    if pred_vp and pred_vp in valid_viewpoints:
                        effective_prediction.append(pred_vp)
                        thought_list.append(pred_thought)
                        logger.info(f"Valid prediction: {pred_vp}")
                        break 
                    else:
                        logger.warning(f"Invalid viewpoint: {raw_pred_vp}. Valid options: {valid_viewpoints}")
                        
                except Exception as e:
                    logger.error(f"Error parsing prediction: {e}")
            
            if effective_prediction:
                return effective_prediction, thought_list, break_flag, score_text, pred_vp
        
        fallback_vp = random.choice(list(valid_viewpoints))
        logger.error(f"All predictions invalid. Fallback to random viewpoint: {fallback_vp}")
        return [fallback_vp], ["Fallback due to invalid predictions"], False, score_text, pred_vp
    
    # ================================================
    # ===== Decision-making under Loop Condition =====
    # ================================================

    def move_to_next_vp_with_loop(self, logger, current_step, instruction, actions, landmarks, history_traj, match_step_info, match_step_vp, estimation, observation, observe_dict):    
        break_flag = True
        score_text = "" 
        pred_vp = None
        valid_viewpoints = set(map(str, observe_dict.keys())) 
        for i in range(2):
            effective_prediction, thought_list = [], []
            batch_responses = self.llm.gpt_infer(NAVIGATOR_LOOP['system'], 
                                                  NAVIGATOR_LOOP['user'].format(actions, observe_dict.keys(), landmarks, history_traj, match_step_info,
                                                                             match_step_vp, estimation, observation),
                                                  num_output=3)

            for decision_reasoning in batch_responses:
                clean_reasoning = re.sub(r'<think>.*?</think>', '', decision_reasoning, flags=re.DOTALL).strip()

                if "Prediction:" not in clean_reasoning:
                    continue
                    
                logger.info(f"================retry id {i} in pred_vp==========")
                logger.info(clean_reasoning)
                
                try:

                    parts = clean_reasoning.split("Prediction:")
                    if len(parts) < 2:
                        raise ValueError("No Prediction found in LLM output.")

                    thought_and_score_text = parts[0].strip()
                    raw_pred_vp = parts[1].strip()

                    if "Score:" in thought_and_score_text:
                        thought_text, score_text = thought_and_score_text.split("Score:", 1)
                        pred_thought = thought_text.strip()
                        score_text = score_text.strip()
                    else:
                        pred_thought = thought_and_score_text
                        score_text = ""
                    pred_vp = re.sub(r'[^0-9]', '', raw_pred_vp)
                    if pred_vp and pred_vp in valid_viewpoints:
                        effective_prediction.append(pred_vp)
                        thought_list.append(pred_thought)
                        logger.info(f"Valid prediction: {pred_vp}")
                        break 
                    else:
                        logger.warning(f"Invalid viewpoint: {raw_pred_vp}. Valid options: {valid_viewpoints}")
                        
                except Exception as e:
                    logger.error(f"Error parsing prediction: {e}")

            if effective_prediction:
                return effective_prediction, thought_list, break_flag, score_text, pred_vp
        
        fallback_vp = random.choice(list(valid_viewpoints))
        logger.error(f"All predictions invalid. Fallback to random viewpoint: {fallback_vp}")
        return [fallback_vp], ["Fallback due to invalid predictions"], False, score_text, pred_vp
    
    # =========================
    # ===== Test Decision =====
    # =========================

    def thought_fusion(self, logger, predictions, thoughts):
        matched_dict = dict()
        for pred, thought in zip(predictions, thoughts):
            if pred not in matched_dict.keys():
                matched_dict[pred] = []
            clean_thought = re.sub(r'<think>.*?</think>', '', thought, flags=re.DOTALL).strip() if '<think>' in thought else thought
            matched_dict[pred].append(clean_thought)
            
        for key, value in matched_dict.items():
            multiple_thoughts = "; ".join(["Thought "+str(idx+1)+": "+thought for idx, thought in enumerate(value)])
            one_thought_response = self.llm.gpt_infer(THOUGHT_FUSION['system'], THOUGHT_FUSION['user'].format(multiple_thoughts))
            one_thought = re.sub(r'<think>.*?</think>', '', one_thought_response, flags=re.DOTALL).strip() if '<think>' in one_thought_response else one_thought_response # type: ignore
            logger.info(f"Pred viewpoint ID: {key} Fused Thought: {one_thought}")
            matched_dict[key] = one_thought
        return matched_dict

    def test_decisions(self, logger, fused_pred_thought, observation, instruction, error_number, observe_dict):
        try:
            cleaned_pred_thought = {}
            for key, value in fused_pred_thought.items():
                clean_value = re.sub(r'<think>.*?</think>', '', value, flags=re.DOTALL).strip() if '<think>' in value else value
                cleaned_pred_thought[key] = clean_value

            for fused_key in list(cleaned_pred_thought.keys()):
                if len(fused_key) > 2:
                    cleaned_pred_thought.pop(fused_key)
                    
            if not cleaned_pred_thought:
                raise ValueError("Error in fused_thought key")
                
            if len(cleaned_pred_thought.keys()) == 1:
                for key, value in cleaned_pred_thought.items():
                    return key, value, error_number
            else:
                fused_pred_thought_ = "; ".join(["Direction Viewpoint ID: "+key+" Thought: "+value for key, value in cleaned_pred_thought.items()])
                
                for i in range(2): 
                    logger.info(f"========== {i} retry in test decision==========")
                    next_vp_response = self.llm.gpt_infer(DECISION_TEST['system'], DECISION_TEST['user'].format(cleaned_pred_thought.keys(), observation, instruction, fused_pred_thought_))
                    next_vp = re.sub(r'<think>.*?</think>', '', next_vp_response, flags=re.DOTALL).strip() if '<think>' in next_vp_response else next_vp_response # type: ignore
                    logger.info(f"Next predicted action is {next_vp}")
                    if re.search(r'\D', next_vp):
                        next_vp = re.search(r'\d+', next_vp).group() 
        
            logger.info(f"In test decision the predicted direction: {next_vp}")
            logger.info(f"In test decision the predicted thought: {cleaned_pred_thought[next_vp]}")
            return next_vp, cleaned_pred_thought[next_vp], error_number
        except Exception as e:
            logger.info(f"Error in test decision {e}")
            error_number += 1
            logger.info(f"Error number is {error_number}")
            
            if error_number >= 2: 
                error_number = 0 
                if cleaned_pred_thought and all(len(key) < 2 for key in cleaned_pred_thought):
                    logger.info(f"Random choice a next predicted action {next_vp} in cleaned_pred_thought, error number reset to {error_number}")
                    next_vp, _ = random.choice(list(cleaned_pred_thought.items()))
                    return next_vp, cleaned_pred_thought[next_vp], error_number
                else:
                    next_vp, observe_description = random.choice(list(observe_dict.items()))
                    logger.info(f"Random choice a next predicted action {next_vp}, error number reset to {error_number}")
                    return next_vp, observe_description, error_number
            return "error_next_vp", "None", error_number
        
    
    def make_decisions(self, logger, fused_pred_thought, observation, instruction, error_number, observe_dict):
        try:
            # Clean input thoughts first
            cleaned_pred_thought = {}
            for key, value in fused_pred_thought.items():
                clean_value = re.sub(r'<think>.*?</think>', '', value, flags=re.DOTALL).strip() if '<think>' in value else value
                cleaned_pred_thought[key] = clean_value

            for fused_key in list(cleaned_pred_thought.keys()):
                if len(fused_key) > 2:
                    cleaned_pred_thought.pop(fused_key)
                    
            if not cleaned_pred_thought:
                raise ValueError("Error in fused_thought key")
                
            if len(cleaned_pred_thought.keys()) == 1:
                for key, value in cleaned_pred_thought.items():
                    return key, value, error_number
            else:
                fused_pred_thought_ = "; ".join(["Direction Viewpoint ID: "+key+" Thought: "+value for key, value in cleaned_pred_thought.items()])
                
                for i in range(2): 
                    logger.info(f"========== {i} retry in test decision==========")
                    next_vp_response = self.llm.gpt_infer(DECISION_TEST['system'], DECISION_TEST['user'].format(cleaned_pred_thought.keys(), observation, instruction, fused_pred_thought_))
                    # Clean the response
                    next_vp = re.sub(r'<think>.*?</think>', '', next_vp_response, flags=re.DOTALL).strip() if '<think>' in next_vp_response else next_vp_response
                    logger.info(f"Next predicted action is {next_vp}")
                    if re.search(r'\D', next_vp):
                        next_vp = re.search(r'\d+', next_vp).group() 
        
            logger.info(f"In test decision the predicted direction: {next_vp}")
            logger.info(f"In test decision the predicted thought: {cleaned_pred_thought[next_vp]}")
            return next_vp, cleaned_pred_thought[next_vp], error_number
        except Exception as e:
            logger.info(f"Error in test decision {e}")
            error_number += 1
            logger.info(f"Error number is {error_number}")
            
            if error_number >= 2: 
                error_number = 0 
                if cleaned_pred_thought and all(len(key) < 2 for key in cleaned_pred_thought):
                    logger.info(f"Random choice a next predicted action {next_vp} in cleaned_pred_thought, error number reset to {error_number}")
                    next_vp, _ = random.choice(list(cleaned_pred_thought.items()))
                    return next_vp, cleaned_pred_thought[next_vp], error_number
                else:
                    next_vp, observe_description = random.choice(list(observe_dict.items()))
                    logger.info(f"Random choice a next predicted action {next_vp}, error number reset to {error_number}")
                    return next_vp, observe_description, error_number
            return "error_next_vp", "None", error_number
        
    def angular_distance(self, h1, h2):

        return np.abs(np.arctan2(np.sin(h1 - h2), np.cos(h1 - h2)))
    

    # =========================
    # ===== Loop Detection =====
    # ========================= 

    def detect_latest_loop_with_heading(self, logger, data, dist_thresh=0.25, angle_thresh=np.pi/6, window_size=20, eps=1e-3):
            if len(data) < 2:
                return False, None

            i = len(data) - 1
            curr = data[i]
            curr_pos = np.array(curr["position"])
            curr_heading = curr["heading"]

            best_match = None
            min_score = float("inf")

            w_dist = 0.5
            w_angle = 0.5
            angle_thresh_deg = np.degrees(angle_thresh)

            for j in range(max(0, i - window_size), i):
                if i - j <= 1:
                    continue

                prev = data[j]
                prev_pos = np.array(prev["position"])
                prev_heading = prev["heading"]

                dist = np.linalg.norm(curr_pos - prev_pos)
                angle_rad = self.angular_distance(curr_heading, prev_heading)
                angle_deg = np.degrees(angle_rad)

                logger.info(f"[LoopCheck] Compare step {i} vs {j} | Δdist={dist:.3f}, Δangle={angle_deg:.1f}°")

                if dist <= dist_thresh and angle_rad <= angle_thresh + eps:
                    norm_dist = dist / dist_thresh
                    norm_angle = angle_deg / angle_thresh_deg

                    score = w_dist * norm_dist + w_angle * norm_angle

                    if score < min_score:
                        min_score = score
                        best_match = {
                            "step": i,
                            "position": curr["position"],
                            "matched_step": j,
                            "distance": round(dist, 3),
                            "angle_diff": round(angle_deg, 1)
                        }

            if best_match is not None:
                return True, best_match
            else:
                return False, None


    

    def extract_step_info(self, logger, navigation_history, loop_info, position_history) -> tuple:
        
        matched_step = loop_info.get("matched_step")
        if matched_step is None:
            raise ValueError("loop_info 中未包含 matched_step 字段")

        pattern = re.compile(
            rf"-> Step {matched_step} Observation:\s*(.*?)\s*Thought:\s*Thought:\s*(.*?)(?=-> Step \d+|$)",
            re.DOTALL
        )
        match = pattern.search(navigation_history)
        if not match:
            raise ValueError(f"未找到 Step {matched_step} 的信息")

        observation = match.group(1).strip()
        thought = match.group(2).strip()

        info = {
            "observation": observation,
            "thought": thought
        }
        pred_vp = None
        for step_data in position_history:
            if step_data.get("step") == matched_step:
                pred_vp = step_data.get("pred_vp")
                break

        if pred_vp is None:
            raise ValueError(f"未在 position_history 中找到 step == {matched_step} 对应的 pred_vp")

        logger.info(f"loop_info: {info}, pred_vp: {pred_vp}")

        return info, pred_vp


    
    def compare_matched_visual_similarity(
    self,
    logger,
    navigation_history: list,
    loop_info: dict,
    position_history: list,
    clip_sim: CLIPSimilarity,
    episode_id: int,
    current_step: int,
    candidate_direction_ids: set
):
     
        matched_step = loop_info.get("matched_step")
        if not isinstance(matched_step, int):
            raise TypeError("loop_info['matched_step'] should be int type")

        if matched_step is None:
            raise ValueError("loop_info does not contain matched_step field")

        matched_entry = next((item for item in navigation_history if item.get("step") == matched_step), None)
        if matched_entry is None:
            raise ValueError(f"Could not find information for Step {matched_step}")

        observation = matched_entry.get("observation", "").strip()
        thought = matched_entry.get("thought", "").strip()


        matched_info = {
            "observation": observation,
            "thought": thought
        }

        matched_direction_id = None
        for step_data in position_history:
            if step_data.get("step") == matched_step:
                matched_direction_id = step_data.get("pred_vp")
                break

        if matched_direction_id is None:
            raise ValueError(f"Could not find pred_vp for step == {matched_step} in position_history")

        logger.info(f"[Matched] Step {matched_step} => pred_vp: {matched_direction_id}")

        most_similar_dir, scores = clip_sim.compute_similarity(
            episode_id=episode_id,
            matched_step=matched_step,
            matched_direction_id=matched_direction_id,
            current_step=current_step,
            candidate_direction_ids=candidate_direction_ids
        )
        logger.info(f"[CLIP Match] Most similar dir_id: {most_similar_dir}")
        logger.info(f"[CLIP Match] All scores: {scores}")

        return matched_info, matched_direction_id, most_similar_dir, scores


    def compute_relative_movements(self, logger, position_history):

        movements = [] 

        for i in range(1, len(position_history)):
            prev = position_history[i - 1]
            curr = position_history[i]

            pos_prev = np.array(prev["position"])
            pos_curr = np.array(curr["position"])
            dist = np.linalg.norm(pos_curr - pos_prev)

            heading_prev = prev["heading"]
            heading_curr = curr["heading"]
            angle_diff_rad = heading_curr - heading_prev
            angle_diff_deg = (angle_diff_rad * 180.0 / np.pi + 180) % 360 - 180

            movement_str = f"Step {prev['step']} → Step {curr['step']}: Moved {dist:.2f} meters, turned {angle_diff_deg:.1f}°"
            movements.append(movement_str)
            logger.info(f"Movement at step {curr['step']}: {movement_str}")

        return "\n".join(movements)



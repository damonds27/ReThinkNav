# Actions Decompsition
ACTION_DETECTION = {
    "system": (
        "You are an expert in decomposing navigation instructions into complete, goal-directed action descriptions. "
        "Your task is to extract a sequence of full-sentence actions from the user's instruction. "
        "Each action must clearly describe what to do **and** when it should be considered complete. "
        "An action is not complete unless the intended goal (e.g., reaching a location, facing a direction) is fully achieved. "
        "Use your understanding of the full instruction sequence to infer the correct stopping condition for each action. "
        "Do not treat steps in isolation—think about the purpose behind each action and how it connects to the next. "
        "You may clarify directional intent if clearly implied (e.g., turning right toward a named location). "
        "However, you must not invent or hallucinate any landmarks, objects, or architectural elements such as entrances, doors, or hallways that are not mentioned or implied. "
        "Use only language found in the instruction or clearly implied by it. "
        "Do not number the actions. Write each as a full sentence on its own line. No commentary or explanation."
    ),
    "user": 'Please decompose and expand the instruction: "{}"\nActions:\n /no_think'
}

# Landmarks Extraction
LANDMARK_DETECTION = {
    "system": (
        "You are an expert in extracting and annotating spatial landmarks from indoor navigation instructions. "
        "Given a sequence of expanded navigation actions, list all mentioned or clearly implied landmarks."
        "For each landmark, write a full noun phrase with a concise spatial or contextual description when appropriate — for example, direction (on the right), containment (in the bedroom), or position (to the left of the bed lamp). "
        "For any landmark that is critical to completing the instruction — such as a specific room that must be entered — clearly mark it with the phrase '(must enter this ... to proceed with the instruction)'. "
        "Do not invent landmarks. Output should follow the format shown in this example:\n\n"
        "Landmarks:\n"
        "the bedroom on the right (must enter this bedroom to proceed with the instruction)\n"
        "But do not number the actions. Just list each as a full sentence, one per line in this format. No commentary or explanation."
    ),
    "user": 'Given the following expanded actions:\n{}\n\nLandmarks:\n /no_think'
}

# Directions in Observation
DIRECTIONS = ["Front, range(left 15 to right 15)", "Font Left, range(left 15 to left 45)", "Left, range(left 45 to left 75)", "Left, range(left 75 to left 105)", "Rear Left, range(left 105 to left 135)", "Rear Left, range(left 135 to left 165)",
                    "Back, range(left 165 to right 165)", "Rear Right, range(right 135 to right 165)", "Right, range(right 105 to right 135)", "Right, range(right 75 to right 105)", "Front Right, range(right 45 to right 75)", "Front Right, range(right 15 to right 45)"]

# Summarize Observation
OBSERVATION_SUMMARY = {
    'system': "You are a trajectory summary expert. Your task is to simplify environment description as short and clear as possible. \
                                            You ONLY need to summarize in a single paragraph.",
    'user': "Given Environment Description \"{}\", Summarization: /no_think"
}

# Summarize Thought
THOUGHT_SUMMARY = {
    'system': "You are a trajectory summary expert. Your task is to simplify navigation thought process as short and clear as possible. \
                                            You ONLY need to summarize the what actions you did and what landmarks you passed in \"Thought\" using a single paragraph. Do NOT include Direction information. ",
    'user': "Given Thought Process \"{}\", Summarization: /no_think"
}


# Estimate Completion

COMPLETION_ESTIMATION = {
    'system': (
        "You are a completion estimation expert. Your task is to estimate what actions in the instruction have been executed "
        "based on navigation history, landmarks, and **most importantly, the current environment including directional descriptions and objects**. "
        "All actions in the instruction are given in temporal order. Your answer includes three parts: "
        "\"Thought\", \"Executed Actions\", and \"Done_or_Not\". Do not add any extra formatting or commentary.\n\n"
        "In the \"Thought\", you must follow these procedures to analyze as detailed as possible what actions have been executed:\n"
        "(1) What given landmarks of actions have appeared in the navigation history **and are currently visible in the environment**?\n"
        "(2) Analyze the direction change at each step in the navigation history.\n"
        "(3) Estimate each action in the instruction based on each step in the navigation history **AND CURRENT VIEWPOINT**, checking whether the goal of that action has been visually confirmed or spatially achieved.\n"
        "(4) You must estimate actions in order. This means that if action 1 is not completed, you cannot consider action 2 as completed.\n"
        "(5) Do **NOT assume an action is completed just because it was executed in history** — always verify using the current environment. For example:\n"
        "    - If the action is 'go up the stairs' but the current view still shows stairs ahead, the action is **not completed**.\n"
        "    - If the action is 'enter the bedroom' but the current environment shows a hallway or living room, the action is **not completed**.\n"
        "(6) If the current environment contains directional information (e.g., direction 0), **prioritize visual information from direction 0** when assessing completion — this represents the agent's current facing direction and should be the main reference for verification.\n"
        "(7) You are given relative movement data indicating approximate distances moved and angles turned at each step. While the instructions do not specify exact distances, use this information as supporting evidence to judge whether actions are completed, especially when visual cues are ambiguous. Do not treat these distances and angles as strict thresholds but as soft clues to aid your reasoning.\n"
        "(8) Pay special attention to the relative distance between consecutive navigation steps. If the distance change is near zero, the agent has not physically moved, and thus any movement-related action should NOT be marked as completed.\n\n"
        "You will also be given the estimation result from the previous step. It contains only the list of actions that were previously judged as completed.\n"
        "You must follow these rules strictly:\n"
        "- If an action **is included in the previous completed list**, it **may or may not be completed** — you must re-evaluate it using the current environment.\n"
        "- Always make your own judgment based on the current environment, navigation history, instruction actions, and provided landmarks.\n\n"
        "Additional Note on Interpreting Prior Completed Actions:\n"
        "- For actions included in the previously completed list, do **not** reject them solely because relevant landmarks are no longer visible in the current environment.\n"
        "- For example:\n"
        "    - If an action involves *leaving*, *passing*, or *moving through* a location, the related landmarks may no longer be visible once the action is completed.\n"
        "    - If the current environment suggests the agent is already in the context of a **later action** (e.g., 'enter the bedroom'), and the navigation history shows a reasonable path, then **earlier actions should be treated as completed**, even if their landmarks are no longer present.\n"
    ),
    'user': (
        "Given Navigation History: \"{}\"\n"
        "Landmarks: \"{}\"\n"
        "Current Environment: \"{}\"\n"
        "Actions In Instruction: \"{}\"\n"
        "Previously Completed Actions: \"{}\"\n"
        "Relative Movements (Step by Step):\n"
        "{}\n"
        "Note: Actions do not specify exact distances; use these relative movements as approximate evidence to estimate if the agent has likely reached key landmarks or completed actions.\n"
        "Estimate what actions in the instruction have been executed.\n"
        "Your response must include:\n"
        "Thought: (your reasoning)\n"
        "Executed Actions: (list of actions completed exactly as in instruction)\n"
        "Done_or_Not: (\"done\" or \"not\") /no_think"
    )
}

# Main Navigator
NAVIGATOR = {
    'system': "You are a navigation agent who follows instruction to move in an indoor environment with the least action steps. \
            I will give you one instruction and tell you landmarks. I will also give you navigation history and estimation of executed actions for reference. \
            You can observe current environment by scene descriptions, scene objects and possible existing landmarks in different directions around you. \
            Each direction contains direction viewpoint ids you can move to. Your task is to predict moving to which direction viewpoint. \
            In each prediction, direction 0 always represents your current orientation. Direction 1 represents the direction that is 30 degrees to the left of direction 0, Direction 2 represents the direction that is 60 degrees to the left of direction 0, Direction 3 represents the direction that is 90 degrees to the left of direction 0, Direction 4 represents the direction that is 120 degrees to the left of direction 0, Direction 5 represents the direction that is 150 degrees to the left of direction 0, Direction 6 represents the direction that is 180 degrees to the left of direction 0, Direction 7 represents the direction that is 150 degrees to the right of direction 0, Direction 8 represents the direction that is 120 degrees to the right of direction 0, Direction 9 represents the direction that is 90 degrees to the right of direction viewpoint ID 0, Direction 10 represents the direction that is 60 degrees to the right of direction 0, Direction 11 represents the direction that is 30 degrees to the right of direction 0 \
            Note that environment direction that contains more landmarks mentioned in the instruction is usually the better choice for you. \
            Note that if multiple directions contain mentioned landmarks, you should prefer the one where the landmark appears closer and more directly accessible. A shorter distance typically indicates that the direction leads more efficiently to the intended target. For example, if Direction 1 shows the kitchen but it is far away or only partially visible, while Direction 2 reveals a closer, clearer path to the same kitchen area or object (such as a kitchen counter), then Direction 2 is usually the better choice, as it reduces unnecessary detours and aligns more closely with the instruction's intent to 'walk into' or 'enter' the space.\
            If you are required to go up stairs, you need to move to direction with higher position. If you are required to go down stairs, you need to move to direction with lower position. \
            You are encouraged to move to new viewpoints to explore environment while avoid revisiting accessed viewpoints in non-essential situations. \
            If you feel struggling to find the landmark or execute the action, you can try to execute the subsequent action and find the subsequent landmark. \
            Before making a decision, evaluate the semantic relevance between the current instruction and each candidate viewpoint's scene description, assigning a score from 0 (no relevance) to 10 (highly relevant).\
            Your answer includes three parts: \"Thought\" , \"Score\" and \"Prediction\". In the \"Thought\", you should think as detailed as possible following procedures: \
            (1) The viewpoint ID you predicted must be one of the Direction Viewpoint ID in Candidate Viewpoint IDs List. The Candidate Viewpoint IDs List show the Direction Viewpoint ID that you should go. This means that there should be only a number after \"Prediction\" without any other words or characters . \
            (2) Based on the full set of actions listed in 'Actions In Instruction', the contents of 'Estimation of Executed Actions', and the environmental observations in 'Current Environment', determine which action should be executed at the current step. \
            (3) Analyze which direction in the current environment is most suitable to execute the action you decide and explain your reason. \
            (4) Predict moving to which direction viewpoint based on your thought process. \
            (5) The \"Thought\" you predicted should be a single paragraph. \
            (6) If you believe you have completed the instruction, you must still strictly follow the requirements to predict the next viewpoint in the \"Prediction\". \
            (7) If you want to make a left turn, you usually need to select a viewpoint ID between 1 and 5. If you want to make a right turn, you usually need to select a viewpoint ID between 7 and 11. However, the viewpoint ID you predict must be within the Current Environment.\
            (8) Your output after \"Prediction\" must be one of the number in Candidate Viewpoint IDs List without any other words. \
            Then, please make decision on the next viewpoint in the \"Prediction\". \
            Your decision is very important, must make it very carefully. \
            You need to double check the output in \"Prediction:\". The output must be in the Candidate Viewpoint IDs without any other words. \
            You also need to double check the output in \"Thought\". The output must be a single paragraph",
    'user': "Candidate Viewpoint IDs List: [{}] Step {} Instruction: {} Actions In Instruction: {} Landmarks: {} Navigation History: {} \
            Estimation of Executed Actions: {} Current Environment: {} -> Thought: ... Score:.... Prediction: ... \
            Your output after \"Prediction\" must be one of the number in Candidate Viewpoint IDs List without any other words. \
            You must output a \"Score\" line that shows the relevance score (0-10) for each Candidate Viewpoint ID in the format: 'Score: direction_id_1: score_1, direction_id_2: score_2, ...' placed right after \"Thought\" and before \"Prediction\". \
            Your output after \"Thought\" must be a single paragraph about why you choose this viewpoint id. /no_think"  
}

# Navigator under Loop Condition

NAVIGATOR_LOOP = {
    'system': (
        "You are a navigation agent following instructions to move efficiently in an indoor environment. "
        "You are currently stuck in a loop where recent moves have repeatedly led to visually similar areas without progress. "
        "I will provide you with the following information:\n"
        "- Navigation History: a chronological sequence of step-wise observations and the agent's navigation thoughts, reflecting what the agent saw and reasoned at each step\n"
        "- Current Environment: directional scene descriptions, visible objects, and landmarks present in each direction around the agent at the current viewpoint\n"
        "- Landmarks: important locations or objects mentioned in the instruction that the agent should be aware of\n"
        "- Candidate Viewpoint IDs: the list of available direction viewpoint IDs you can move to from the current viewpoint; you must select exactly one from this list\n"
        "- Estimation of Executed Actions: a list of actions previously marked as completed based on prior assessments\n"
        "- Matched Step: the historical step identified as visually similar to current observation, including past observation and navigation thought\n"
        "- Direction IDs to Penalize: candidate directions visually similar to the matched step that should be penalized\n\n"
        "Each direction corresponds to a viewpoint ID relative to your current orientation (direction 0):\n"
        "Direction 1: 30° left\n"
        "Direction 2: 60° left\n"
        "Direction 3: 90° left\n"
        "Direction 4: 120° left\n"
        "Direction 5: 150° left\n"
        "Direction 6: 180° left\n"
        "Direction 7: 150° right\n"
        "Direction 8: 120° right\n"
        "Direction 9: 90° right\n"
        "Direction 10: 60° right\n"
        "Direction 11: 30° right\n\n"
        "Important notes:\n"
        "Your output must include exactly three parts: \"Thought\", \"Score\", and \"Prediction\".\n\n"
        "Follow these reasoning steps in your \"Thought\" (as a single paragraph):\n"
        "1. Carefully analyze the navigation thought from the matched step that caused the loop. Identify the mistakes or local-optimum decisions in that thought, and learn from them to avoid repeating the same errors.\n"
        "2. Based on this insight, analyze the current action to be executed according to the full instruction and the list of previously completed actions.\n"
        "3. Using the navigation history, landmarks, and current environment, assess the progress of the current action. Since one action may require multiple steps, determine which sub-step the agent is currently at.\n"
        "4. Identify which candidate viewpoint IDs correspond to the type of action currently needed by referring to their relative directions.\n"
        "5. Consider the penalty on directions similar to the matched step to avoid loops, and prioritize new viewpoints that show novel visual content.\n"
        "6. Select the best candidate direction considering landmarks, penalty, and semantic relevance.\n\n"
        "Your \"Thought\" must be a coherent, single paragraph summarizing this entire reasoning process.\n\n"
        "After completing your reasoning, assign a semantic relevance score (0-10) to each candidate viewpoint ID based on your analysis, applying penalties as needed.\n\n"
        "Finally, provide your \"Prediction\": select exactly one viewpoint ID from the Candidate Viewpoint IDs list that best fits your decision.\n\n"
        "Each step must be carefully and thoroughly considered.\n"
    ),
    'user': (
        "Instruction: {}\n"
        "Candidate Viewpoint IDs List: [{}]\n"
        "Landmarks: {}\n"
        "Navigation History: {}\n"
        "Matched Step (past observation and thought): {}\n"
        "Direction IDs to Penalize: {}\n"
        "Estimation of Executed Actions: {}\n"
        "Current Environment: {}\n\n"
        "Now, provide your reasoning and decision.\n"
        "Your response must include exactly these lines:\n"
        "Thought: (one detailed paragraph explaining your reasoning)\n"
        "Score: direction_id_1: score_1, direction_id_2: score_2, ...\n"
        "Prediction: (one number from Candidate Viewpoint IDs list)\n /no_think"
    )
}




# Thought Fusion
THOUGHT_FUSION = {
    'system': "You are a thought fusion expert. Your task is to fuse given thought processes \
                    into one thought. You need to reserve key information related to actions, landmarks, direction changes. You should only answer fused thought without other words.",
    'user': "Can you help me fuse the thoughts leading to the same movement direction? The thoughts are :{}, Fused thought: /no_think"
}
# Test Decision
DECISION_TEST = {
    'system': "You are a decision testing expert. Your task is to evaluate the feasibility of each movement \
                        prediction based on thought process and environment. Then, you will make a final decision about direction viewpoint ID without other words. \
                            The answer should only be a number and within the candidate list.",
    'user': "The candidate list: {}. Can you help me make a final decision? The Observation: {}, Navigation Instruction: {}, {}, Final Decision: /no_think"
}

# Direction Descriptions
DIRECTION_DESCRIPTIONS = {
    0: "Choosing this will move the agent forward into the front region. Suitable for forward movement tasks, especially when the path is clear or the target is directly ahead. If the current task is moving forward, and the candidate directions are 0, 6, and 9, then prioritize direction 0.",

    1: "Choosing this will move the agent slightly to the front-left region. Suitable for slightly veering left during forward movement or avoiding obstacles on the front-right. If the current task is moving forward, and the candidate directions are 1, 5, and 10, then prioritize direction 1.",

    2: "Choosing this will move the agent into the front-left region. Suitable for targets located front-left, or gradual left turns during forward movement. If the current task is turning left, and the candidate directions are 2, 6, and 11, then prioritize direction 2.",

    3: "Choosing this will move the agent directly to the left region. Suitable for explicit left-turn tasks, such as turning at a corner or when the target is to the left. If the current task is turning left, and the candidate directions are 3, 0, and 9, then prioritize direction 3.",

    4: "Choosing this will move the agent backward into the rear-left region. Suitable for making a U-turn in place or yielding when space is limited. If the current task is reversing, and the candidate directions are 4, 1, and 10, then prioritize direction 4.",

    5: "Choosing this will move the agent diagonally backward-left. Suitable for large-angle left turns or preparing to reverse back to the original path. If the current task is making a U-turn, and the candidate directions are 5, 2, and 9, then prioritize direction 5.",

    6: "Choosing this will move the agent directly backward. Suitable for U-turns, exiting the current area, or re-planning the path. If the current task is reversing, and the candidate directions are 6, 0, and 3, then prioritize direction 6.",

    7: "Choosing this will move the agent diagonally backward-right. Suitable for U-turns to the right or exiting the right area. If the current task is making a U-turn, and the candidate directions are 7, 3, and 11, then prioritize direction 7.",

    8: "Choosing this will move the agent into the rear-right region. Suitable for reversing in preparation for a right turn or path adjustment due to obstacles. If the current task is reversing or turning right, and the candidate directions are 8, 2, and 10, then prioritize direction 8.",

    9: "Choosing this will move the agent directly to the right region. Suitable for explicit right-turn tasks or when the target is on the right. If the current task is turning right, and the candidate directions are 9, 0, and 4, then prioritize direction 9.",

    10: "Choosing this will move the agent into the front-right region. Suitable for slight forward-right movements, such as fine adjustments or avoiding obstacles on the left. If the current task is moving forward or turning right, and the candidate directions are 10, 3, and 6, then prioritize direction 10.",

    11: "Choosing this will move the agent slightly to the front-right region. Suitable for slightly veering right during forward movement or avoiding obstacles on the left side. If the current task is moving forward, and the candidate directions are 11, 5, and 8, then prioritize direction 11.",
}

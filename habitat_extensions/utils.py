from typing import Dict

import numpy as np
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps as habitat_maps
from habitat.utils.visualizations.utils import draw_collision

from habitat_extensions import maps

cv2 = try_cv2_import()


def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    observation_size = -1
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view.append(rgb)

    # draw depth map if observation has depth info. resize to rgb size.
    if "depth" in observation:
        if observation_size == -1:
            observation_size = observation["depth"].shape[0]
        depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        depth_map = cv2.resize(
            depth_map,
            dsize=(observation_size, observation_size),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(depth_map)

    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    map_k = None
    if "top_down_map_vlnce" in info:
        map_k = "top_down_map_vlnce"
    elif "top_down_map" in info:
        map_k = "top_down_map"

    if map_k is not None:
        td_map = info[map_k]["map"]

        td_map = maps.colorize_topdown_map(
            td_map,
            info[map_k]["fog_of_war_mask"],
            fog_of_war_desat_amount=0.75,
        )
        td_map = habitat_maps.draw_agent(
            image=td_map,
            agent_center_coord=info[map_k]["agent_map_coord"],
            agent_rotation=info[map_k]["agent_angle"],
            agent_radius_px=min(td_map.shape[0:2]) // 24,
        )
        if td_map.shape[1] < td_map.shape[0]:
            td_map = np.rot90(td_map, 1)

        if td_map.shape[0] > td_map.shape[1]:
            td_map = np.rot90(td_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = td_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        td_map = cv2.resize(
            td_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((egocentric_view, td_map), axis=1)
    return frame


# def extract_topdown_map(info: Dict, save_path: str = None) -> np.ndarray:
#     """
#     提取并保存 top-down map（旋转后的高分辨率版本）。
    
#     Args:
#         info: step() 返回的 info dict.
#         save_path: 如果给定路径，就把图像保存下来.

#     Returns:
#         处理后的 top-down map (np.ndarray).
#     """
#     map_k = None
#     if "top_down_map_vlnce" in info:
#         map_k = "top_down_map_vlnce"
#     elif "top_down_map" in info:
#         map_k = "top_down_map"
#     else:
#         return None  # 没有 top-down map

#     td_map = info[map_k]["map"]

#     # 上色 & 绘制 agent
#     td_map = maps.colorize_topdown_map(
#         td_map,
#         info[map_k]["fog_of_war_mask"],
#         fog_of_war_desat_amount=0.75,
#     )
#     td_map = habitat_maps.draw_agent(
#         image=td_map,
#         agent_center_coord=info[map_k]["agent_map_coord"],
#         agent_rotation=info[map_k]["agent_angle"],
#         agent_radius_px=min(td_map.shape[0:2]) // 24,
#     )

#     # 旋转，确保是横向
#     if td_map.shape[1] < td_map.shape[0]:
#         td_map = np.rot90(td_map, 1)
#     if td_map.shape[0] > td_map.shape[1]:
#         td_map = np.rot90(td_map, 1)

#     # ✅ 保持原始分辨率，不缩放
#     if save_path is not None:
#         cv2.imwrite(save_path, cv2.cvtColor(td_map, cv2.COLOR_RGB2BGR))

#     return td_map


# def observations_to_image(
#     observation: Dict,
#     info: Dict,
#     save_topdown_map: str = None  # 新增参数，可选保存单独的高分辨率地图
# ) -> np.ndarray:
#     r"""Generate image of single frame from observation and info
#     returned from a single environment step().

#     Args:
#         observation: observation returned from an environment step().
#         info: info returned from an environment step().
#         save_topdown_map: 如果给定路径，就把旋转后的高分辨率 top-down map 单独保存下来.

#     Returns:
#         generated image of a single frame.
#     """
#     egocentric_view = []
#     observation_size = -1
#     if "rgb" in observation:
#         observation_size = observation["rgb"].shape[0]
#         rgb = observation["rgb"][:, :, :3]
#         egocentric_view.append(rgb)

#     # draw depth map if observation has depth info. resize to rgb size.
#     if "depth" in observation:
#         if observation_size == -1:
#             observation_size = observation["depth"].shape[0]
#         depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
#         depth_map = np.stack([depth_map for _ in range(3)], axis=2)
#         depth_map = cv2.resize(
#             depth_map,
#             dsize=(observation_size, observation_size),
#             interpolation=cv2.INTER_CUBIC,
#         )
#         egocentric_view.append(depth_map)

#     assert (
#         len(egocentric_view) > 0
#     ), "Expected at least one visual sensor enabled."
#     egocentric_view = np.concatenate(egocentric_view, axis=1)

#     # draw collision
#     if "collisions" in info and info["collisions"]["is_collision"]:
#         egocentric_view = draw_collision(egocentric_view)

#     frame = egocentric_view

#     map_k = None
#     if "top_down_map_vlnce" in info:
#         map_k = "top_down_map_vlnce"
#     elif "top_down_map" in info:
#         map_k = "top_down_map"
#     # if "top_down_map" in info:
#     #      map_k = "top_down_map"

#     if map_k is not None:
#         td_map = info[map_k]["map"]

#         # 上色 & 绘制 agent
#         td_map = maps.colorize_topdown_map(
#             td_map,
#             info[map_k]["fog_of_war_mask"],
#             fog_of_war_desat_amount=0.75,
#         )
#         td_map = habitat_maps.draw_agent(
#             image=td_map,
#             agent_center_coord=info[map_k]["agent_map_coord"],
#             agent_rotation=info[map_k]["agent_angle"],
#             agent_radius_px=min(td_map.shape[0:2]) // 24,
#         )

#         # 旋转，确保是横向
#         if td_map.shape[1] < td_map.shape[0]:
#             td_map = np.rot90(td_map, 1)
#         if td_map.shape[0] > td_map.shape[1]:
#             td_map = np.rot90(td_map, 1)

#         # ✅ 如果需要保存高分辨率单独地图，就直接保存原始旋转后的版本
#         if save_topdown_map is not None:
#             cv2.imwrite(save_topdown_map, cv2.cvtColor(td_map, cv2.COLOR_RGB2BGR))

#         # 再缩放一份地图，用来拼接到 egocentric_view
#         old_h, old_w, _ = td_map.shape
#         top_down_height = observation_size
#         top_down_width = int(float(top_down_height) / old_h * old_w)
#         td_map_resized = cv2.resize(
#             td_map,
#             (top_down_width, top_down_height),
#             interpolation=cv2.INTER_CUBIC,
#         )
#         frame = np.concatenate((egocentric_view, td_map_resized), axis=1)

#     return frame

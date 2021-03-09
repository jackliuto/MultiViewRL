# helper code from https://github.com/allenai/cordial-sync/blob/master/utils/visualization_utils.py#L109

import numpy as np

class ThorPositionTo2DFrameTranslator(object):
    def __init__(self, frame_shape, cam_position, orth_size):
        self.frame_shape = frame_shape
        self.lower_left = np.array((cam_position[0], cam_position[2])) - orth_size
        self.span = 2 * orth_size

    def __call__(self, position):
        if len(position) == 3:
            x, _, z = position
        else:
            x, z = position

        camera_position = (np.array((x, z)) - self.lower_left) / self.span
        return np.array(
            (
                round(self.frame_shape[0] * (1.0 - camera_position[1])),
                round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
        )

def position_to_tuple(position):
    if "position" in position:
        position = position["position"]
    return (position["x"], position["y"], position["z"])


def get_agent_map_data(env):
    env.step({"action": "ToggleMapView", "agentId": 0})
    cam_position = env.last_event.metadata["cameraPosition"]
    cam_orth_size = env.last_event.metadata["cameraOrthSize"]
    pos_translator = ThorPositionTo2DFrameTranslator(
        env.last_event.events[0].frame.shape,
        position_to_tuple(cam_position),
        cam_orth_size,
    )
    to_return = {
        "frame": env.last_event.events[0].frame,
        "cam_position": cam_position,
        "cam_orth_size": cam_orth_size,
        "pos_translator": pos_translator,
    }
    env.step({"action": "ToggleMapView", "agentId": 0})
    return to_return

def convert2darknet(item_dict,object_types,imgw=300.0,imgh=300.0,n=0):
    out_str = ""
    for k,v in item_dict.items():
        obj_name = k.split('|')[0]
        if obj_name in object_types:
            class_id = object_types.index(obj_name)
            x = (v[0] + v[2])/2/imgw
            y = (v[1] + v[3])/2/imgh
            dw = (v[2] - v[0])/imgw
            dh = (v[3] - v[1])/imgh
            if dw*dh > 0.5e-3:            
                out_str += f"{class_id} {x} {y} {dw} {dh}\n"
    return out_str         
            
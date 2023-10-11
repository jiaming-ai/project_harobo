import json
from typing import Dict, Tuple

from home_robot.core.interfaces import Observations
from home_robot.perception.constants import RearrangeDETICCategories
import numpy as np

SYNONYMS = {
    "bed": "bed",
    "bench": "bench",
    "cabinet": "cupboard closet cabinet",
    "chair": "chair",
    "chest_of_drawers": "dresser chest-of-drawers",
    "couch": "sofa couch",
    "counter": "countertop worktop counter",
    "filing_cabinet": "file-cabinet file-drawer filing-cabinet",
    "hamper": "basket hamper",
    "serving_cart": "cart serving-cart",
    "shelves": "shelving racks shelves",
    "shoe_rack": "shoe-rack shoe-shelf",
    "sink": "basin washbasin sink",
    "stand": "stand",
    "stool": "stool",
    "table": "table",
    "toilet": "toilet",
    "trunk": "chest trunk",
    "wardrobe": "closet cabinet wardrobe",
    "washer_dryer": "washing-machine laundry-machine washer-dryer",
}

SYNONYMS_GENERAL = {
    "bed": "bed",
    "bench": "bench",
    "cabinet": "cupboard closet cabinet",
    "chair": "chair",
    "chest_of_drawers": "dresser chest-of-drawers drawer storage",
    "couch": "sofa couch",
    "counter": "countertop worktop counter table-top",
    "filing_cabinet": "file-cabinet file-drawer filing-cabinet",
    "hamper": "basket hamper",
    "serving_cart": "cart serving-cart",
    "shelves": "shelving racks shelves",
    "shoe_rack": "shoe-rack shoe-shelf",
    "sink": "basin washbasin sink",
    "stand": "stand",
    "stool": "stool",
    "table": "table",
    "toilet": "toilet",
    "trunk": "chest trunk",
    "wardrobe": "closet cabinet wardrobe",
    "washer_dryer": "washing-machine laundry-machine washer-dryer",
}

def read_category_map_file(
    category_map_file: str,
) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Reads a category map file in JSON and extracts mappings between category names and category IDs.
    These mappings are also present in the episodes file but are extracted to use in a stand-alone manner.
    Returns object and receptacle mappings.
    """
    with open(category_map_file) as f:
        category_map = json.load(f)

    obj_name_to_id_mapping = category_map["obj_category_to_obj_category_id"]
    rec_name_to_id_mapping = category_map["recep_category_to_recep_category_id"]
    obj_id_to_name_mapping = {k: v for v, k in obj_name_to_id_mapping.items()}
    rec_id_to_name_mapping = {k: v for v, k in rec_name_to_id_mapping.items()}

    return obj_id_to_name_mapping, rec_id_to_name_mapping


def build_vocab_from_category_map(
    obj_id_to_name_mapping: Dict[int, str], rec_id_to_name_mapping: Dict[int, str]
) -> RearrangeDETICCategories:
    """
    Build vocabulary from category maps that can be used for semantic sensor and visualizations.
    """
    obj_rec_combined_mapping = {}
    for i in range(len(obj_id_to_name_mapping) + len(rec_id_to_name_mapping)):
        if i < len(obj_id_to_name_mapping):
            obj_rec_combined_mapping[i + 1] = obj_id_to_name_mapping[i]
        else:
            obj_rec_combined_mapping[i + 1] = rec_id_to_name_mapping[
                i - len(obj_id_to_name_mapping)
            ]
    vocabulary = RearrangeDETICCategories(
        obj_rec_combined_mapping, len(obj_id_to_name_mapping)
    )
    return vocabulary


class OvmmPerception:
    """
    Wrapper around DETIC for use in OVMM Agent.
    It performs some preprocessing of observations necessary for OVMM skills.
    It also maintains a list of vocabularies to use in segmentation and can switch between them at runtime.
    """

    def __init__(
        self,
        config,
        gpu_device_id: int = 0,
        verbose: bool = False,
        module="grounded_sam",
    ):
        self.config = config
        self._use_detic_viz = config.ENVIRONMENT.use_detic_viz
        self._detection_module = getattr(config.AGENT, "detection_module", "detic")
        self._vocabularies: Dict[int, RearrangeDETICCategories] = {}
        self._current_vocabulary: RearrangeDETICCategories = None
        self._current_vocabulary_id: int = None
        self.verbose = verbose
        if self._detection_module == "detic":
            from home_robot.perception.detection.detic.detic_perception_harobo import DeticPerception
            # TODO Specify confidence threshold as a parameter
            self._segmentation = DeticPerception(
                vocabulary="custom",
                custom_vocabulary=".",
                sem_gpu_id=gpu_device_id,
            )
        elif self._detection_module == "grounded_sam":
            
            from home_robot.perception.detection.grounded_sam.grounded_sam_harobo import (
                GroundedSAMPerception,
            )

            self._segmentation = GroundedSAMPerception(
                custom_vocabulary=".",
                sem_gpu_id=gpu_device_id,
                verbose=verbose,
            )
        else:
            raise NotImplementedError

        # self._use_probabilistic_mapping = config.AGENT.SEMANTIC_MAP.use_probability_map

    @property
    def current_vocabulary_id(self) -> int:
        return self._current_vocabulary_id

    @property
    def current_vocabulary(self) -> RearrangeDETICCategories:
        return self._current_vocabulary

    def update_vocubulary_list(
        self, vocabulary: RearrangeDETICCategories, vocabulary_id: int
    ):
        """
        Update/insert a given vocabulary for the given ID.
        """
        self._vocabularies[vocabulary_id] = vocabulary

    def set_vocabulary(self, vocabulary_id: int, use_synonyms: int = 0):
        """
        Set given vocabulary ID to be the active vocabulary that the segmentation model uses.
        """
        vocabulary = self._vocabularies[vocabulary_id]
        clip_vocal_list = ["."] + list(vocabulary.goal_id_to_goal_name.values()) + ["other"]
        # synonyms
        if use_synonyms == 1:
            synonyms = SYNONYMS
        elif use_synonyms == 2:
            synonyms = SYNONYMS_GENERAL
        else:
            synonyms = {}
        for i in range(len(clip_vocal_list)):
            if clip_vocal_list[i] in synonyms:
                clip_vocal_list[i] = synonyms[clip_vocal_list[i]]

        self._segmentation.reset_vocab(clip_vocal_list)
        self.vocabulary_name_to_id = {
            name: id for id, name in vocabulary.goal_id_to_goal_name.items()
        }
        self._current_vocabulary = vocabulary
        self._current_vocabulary_id = vocabulary_id


    
    def _process_obs(self, obs: Observations):
        """
        Process observations. Add pointers to objects and other metadata in segmentation mask.
        """
        obs.semantic[obs.semantic == 0] = (
            self._current_vocabulary.num_sem_categories - 1
        )
        obs.task_observations["recep_idx"] = (
            self._current_vocabulary.num_sem_obj_categories + 1
        )
        obs.task_observations["semantic_max_val"] = (
            self._current_vocabulary.num_sem_categories - 1
        )
        if obs.task_observations["start_recep_name"] is not None:
            obs.task_observations["start_recep_goal"] = self.vocabulary_name_to_id[
                obs.task_observations["start_recep_name"]
            ]
        else:
            obs.task_observations["start_recep_name"] = None
        if obs.task_observations["place_recep_name"] is not None:
            obs.task_observations["end_recep_goal"] = self.vocabulary_name_to_id[
                obs.task_observations["place_recep_name"]
            ]
        else:
            obs.task_observations["end_recep_name"] = None
        if obs.task_observations["object_name"] is not None:
            obs.task_observations["object_goal"] = self.vocabulary_name_to_id[
                obs.task_observations["object_name"]
            ]
        else:
            obs.task_observations["object_goal"] = None

    def __call__(self, obs: Observations) -> Observations:
        return self.forward(obs)

    # def predict(self, obs: Observations, depth_threshold: float = 0.5) -> Observations:
    #     """Run with no postprocessing. Updates observation to add semantics."""
    #     return self._segmentation.predict(
    #         obs,
    #         depth_threshold=depth_threshold,
    #         draw_instance_predictions=self._use_detic_viz,
    #     )

    def forward(self, obs: Observations, depth_threshold: float = 0.5) -> Observations:
        """
        Run segmentation model and preprocess observations for OVMM skills
        """
        ret = self._segmentation.predict(
            obs, depth_threshold=depth_threshold, 
            draw_instance_predictions=self._use_detic_viz,
        )
        obs.task_observations["semantic_frame"] = ret["semantic_frame"]
        obs.task_observations["instance_map"] = ret["instance_map"]
        obs.task_observations["instance_classes"] = ret["instance_classes"]
        obs.task_observations["instance_scores"] = ret["instance_scores"]
        obs.task_observations["masks"] = ret["masks"]

        if not self.config.GROUND_TRUTH_SEMANTICS:
            obs.semantic = ret['semantic']
            self._process_obs(obs)
            
        else:
            instance_classes = np.array([
                obs.task_observations["object_goal"],
                obs.task_observations["start_recep_goal"],
                obs.task_observations["end_recep_goal"]
                ]) # corresponds to object, start_recep, end_recep
            instance_scores = np.array([1,1,1])
            masks = np.zeros((3,*obs.semantic.shape))

            masks[0] = obs.semantic==obs.task_observations["object_goal"]
            if "start_recep_goal" in obs.task_observations:
                masks[1] = obs.semantic==obs.task_observations["start_recep_goal"]
            if "end_recep_goal" in obs.task_observations:
                masks[2] = obs.semantic==obs.task_observations["end_recep_goal"]

            obs.task_observations["instance_classes"] = instance_classes
            obs.task_observations["instance_scores"] = instance_scores
            obs.task_observations["masks"] = masks
            
        return obs

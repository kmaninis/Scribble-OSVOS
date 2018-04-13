import os

import torch, cv2

from davisinteractive.session import DavisInteractiveSession
from davisinteractive import utils as interactive_utils
from davisinteractive.dataset import Davis

from osvos_scribble import OsvosScribble
from mypath import Path

# General parameters
gpu_id = 2

# Interactive parameters
max_nb_interactions = 5
max_time = None  # Maximum time for each interaction
subset = 'val'
host = 'localhost'  # 'localhost' for subsets train and val.

# OSVOS parameters
time_budget_per_object = 60
parent_model = 'osvos_parent.pth'

save_model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
save_result_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
report_save_dir = save_result_dir

model = OsvosScribble(parent_model, save_model_dir, gpu_id, time_budget_per_object)
davis = Davis(davis_root=Path.db_root_dir())

with DavisInteractiveSession(host='localhost', davis_root=Path.db_root_dir(), subset=subset,
                             report_save_dir=report_save_dir, max_nb_interactions=max_nb_interactions,
                             max_time=max_time) as sess:
    while sess.next():
        # Get the current iteration scribbles
        sequence, scribbles, first_scribble = sess.get_scribbles()
        if first_scribble:
            n_interaction = 1
            n_objects = davis.dataset['sequences'][sequence]['num_objects']
            first_frame = interactive_utils.scribbles.annotated_frames(scribbles)[0]
        else:
            n_interaction += 1
        pred_masks = []
        print('\nRunning sequence {} in interaction {}'.format(sequence, n_interaction))
        for obj_id in range(1, n_objects+1):
            model.train(first_frame, n_interaction, obj_id, scribbles)
            pred_masks.append(model.test(sequence, n_interaction, obj_id, save_result_dir))

        final_masks = interactive_utils.mask.combine_masks(pred_masks)

        # Submit your prediction
        sess.submit_masks(final_masks)

    # Get the result
    report = sess.get_report()

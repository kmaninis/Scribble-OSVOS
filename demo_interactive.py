import torch, cv2
import os
import timeit

from davisinteractive.session import DavisInteractiveSession
from davisinteractive import utils as interactive_utils
from davisinteractive.dataset import Davis

from osvos_scribble import OSVOSScribble
from mypath import Path


def main():
    # General parameters
    gpu_id = 1

    # Configuration used in the challenges
    max_nb_interactions = 8  # Maximum number of interactions
    max_time_per_interaction = 30  # Maximum time per interaction per object

    # Total time available to interact with a sequence and an initial set of scribbles
    max_time = max_nb_interactions * max_time_per_interaction  # Maximum time per object

    # Interactive parameters
    subset = 'val'
    host = 'localhost'  # 'localhost' for subsets train and val.

    # OSVOS parameters
    time_budget_per_object = 20
    parent_model = 'osvos_parent.pth'
    prev_mask = True  # Use previous mask as no-care area when fine-tuning

    save_model_dir = Path.models_dir()
    report_save_dir = Path.save_root_dir()
    save_result_dir = report_save_dir

    model = OSVOSScribble(parent_model, save_model_dir, gpu_id, time_budget_per_object,
                          save_result_dir=save_result_dir)

    seen_seq = {}
    with DavisInteractiveSession(host=host,
                                 davis_root=Path.db_root_dir(),
                                 subset=subset,
                                 report_save_dir=report_save_dir,
                                 max_nb_interactions=max_nb_interactions,
                                 max_time=max_time) as sess:
        while sess.next():
            t_total = timeit.default_timer()

            # Get the current iteration scribbles
            sequence, scribbles, first_scribble = sess.get_scribbles()
            if first_scribble:
                n_interaction = 1
                n_objects = Davis.dataset[sequence]['num_objects']
                first_frame = interactive_utils.scribbles.annotated_frames(scribbles)[0]
                seen_seq[sequence] = 1 if sequence not in seen_seq.keys() else seen_seq[sequence]+1
            else:
                n_interaction += 1
            pred_masks = []
            print('\nRunning sequence {} in interaction {} and scribble iteration {}'
                  .format(sequence, n_interaction, seen_seq[sequence]))
            for obj_id in range(1, n_objects+1):
                model.train(first_frame, n_interaction, obj_id, scribbles, seen_seq[sequence],
                            subset=subset,
                            use_previous_mask=prev_mask)
                pred_masks.append(model.test(sequence, n_interaction, obj_id,
                                             subset=subset,
                                             scribble_iter=seen_seq[sequence]))

            final_masks = interactive_utils.mask.combine_masks(pred_masks)

            # Submit your prediction
            sess.submit_masks(final_masks)
            t_end = timeit.default_timer()
            print('Total time (training and testing) for single interaction: ' + str(t_end - t_total))

        # Get the DataFrame report
        report = sess.get_report()

        # Get the global summary
        summary = sess.get_global_summary(save_file=os.path.join(report_save_dir, 'summary.json'))


if __name__ == '__main__':
    main()

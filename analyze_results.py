from habitat.utils.visualizations.utils import images_to_video
import os
import cv2
import json
from argparse import ArgumentParser

STAGES = ['nav_to_obj','gaze_obj','pick','nav_to_rec','gaze_rec','place']
DATA_PATH = 'datadump/'

def read_results(exp_name):
    result_dir = os.path.join(DATA_PATH,'results',exp_name)
    result_file = os.path.join(result_dir,'episode_results.json')
    if not os.path.isfile(result_file):
        print(f'No result file found for {exp_name}')
        return None
    else:
        with open(result_file,'r') as f:
            return json.load(f)
    
def compare_results(exp_names):
    all_results = {}
    for exp_name in exp_names:
        all_results[exp_name] = read_results(exp_name)
        
    # print differences
    exp_1 = all_results[exp_names[0]]
    exp_2 = all_results[exp_names[1]]

    for eps in exp_1:
        exp_1_eps = exp_1[eps]
        if eps not in exp_2:
            print(f'{eps} not in {exp_names[1]}')
            continue
        exp_2_eps = exp_2[eps]
        key = 'END.ovmm_find_object_phase_success'
        if exp_1_eps[key] != exp_2_eps[key]:
            print(f'{eps} {key}: {exp_names[0]}-{exp_1_eps[key]}, {exp_names[1]}-{exp_2_eps[key]}')

    return all_results
def make_result_videos(exp_name,prefix='snapshot'):

    
    output_dir = os.path.join(DATA_PATH,'videos',exp_name)
    os.makedirs(output_dir,exist_ok=True)
    imgs_dir = os.path.join(DATA_PATH,'images',exp_name)
    
    all_results = read_results(exp_name)
    
    for eps in os.listdir(imgs_dir):
        full_eps_dir = os.path.join(imgs_dir,eps)
        if eps not in all_results:
            print(f'No results for {eps}')
            continue
        eps_result = all_results[eps]
        succ = str(int(eps_result['END.ovmm_find_object_phase_success'])) + \
                    str(int(eps_result['END.ovmm_pick_object_phase_success'])) + \
                    str(int(eps_result['END.ovmm_find_recep_phase_success']))   + \
                    str(int(eps_result['END.ovmm_place_object_phase_success']))
                
        goal_name = eps_result['goal_name']
        fname = f'{succ}_{goal_name}_{eps}'
        
        if os.path.isdir(full_eps_dir):
            print(f'Processing {eps}')

            try:
                images = []
                shape = None
                cur_stage = 0
                for step in range(1,5000):
                    full_file = os.path.join(full_eps_dir,f'{prefix}_{step:03d}.png')
                    if os.path.isfile(full_file):
                        image = cv2.imread(full_file)
                        image = image[...,[2,1,0]]
                        if shape is None:
                            shape = image.shape
                            
                        if shape != image.shape:
                            images_to_video(images,output_dir,fname+'_stage'+str(cur_stage))
                            images = []
                            cur_stage += 1
                            shape = image.shape
                        
                        images.append(image)
                    else:
                        break
            
                images_to_video(images,output_dir,fname+'_stage'+str(cur_stage))
                
            except KeyboardInterrupt:
                break
            except:
                print(f'Error processing {eps}')
                continue


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--exp_names',type=str,nargs='+',default=['harobo_place_rl','harobo_place_hr'])
    arg_parser.add_argument('--make_videos',action='store_true',default=False)
    
    args = arg_parser.parse_args()
    
    if args.make_videos:
        for exp_name in args.exp_names:
            make_result_videos(exp_name)
    else:
        if len(args.exp_names) > 1:
            compare_results(args.exp_names)
'''
python run.py --ill_rec --save_imgs --method doctr
'''
import cv2
from datetime import datetime
from doctr.inference import get_models, run_model 
import argparse


def capture_camera(model_geo, model_ill, idx_cam, args):
    cam = cv2.VideoCapture(idx_cam, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    do_doc_detection = False

    while(True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        img_name = '%s/%s.png' % (args.path_result, timestamp)
        
        ret, frame = cam.read()
        if not ret:
            break
        
        cv2.imshow('input', frame)

        if do_doc_detection:
            if args.method == 'doctr':
                img_1, img_2 = run_model(frame, img_name, model_geo, model_ill, args)
                if img_1 is not None: cv2.imshow('result_1', img_1)
                if img_2 is not None: cv2.imshow('result_2', img_2)
            
            # if args.method == 2:
            #     ...
                

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):       # q
            break
        if key % 256 == 32:              # SPACE
            if not do_doc_detection:
                do_doc_detection = True
                print('start detecting documents')
                continue
            if do_doc_detection:
                do_doc_detection = False
                print('end detecting documents')
                continue

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--path_data', default='data/')
    parser.add_argument('--path_result', default='data/result_doctr/')
    parser.add_argument('--Seg_path', default='doctr/models/seg.pth')
    parser.add_argument('--GeoTr_path', default='doctr/models/geotr.pth')
    parser.add_argument('--IllTr_path', default='doctr/models/illtr.pth')
    parser.add_argument('--ill_rec', default=False, action="store_true")
    parser.add_argument('--save_imgs', default=False, action="store_true")
    parser.add_argument('--method', default='doctr')
    args = parser.parse_args()

    model_geo, model_ill = get_models(args)
    capture_camera(model_geo, model_ill, 2, args)

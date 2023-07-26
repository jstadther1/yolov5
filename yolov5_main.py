import cv2
import torch
from grabscreen import grab_screen
import time
import numpy as np

def main():

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(torch.device('cuda'))
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='data/best-55-new.pt', force_reload=True).to(torch.device('cuda'))
    frame_count = 0
    start_time = time.time()

    while True:
        reference_aoi = (0, 50, 700, 400)
        reference_grab = grab_screen(region=reference_aoi)
        # model.cuda()
        model.conf = 0.3
        # model.classes = [0, 77]
        results = model([reference_grab])

        img = np.squeeze(results.render())
        # tens = results.pandas().xyxy[0]
        # for _, row in tens.iterrows():
        #     bbox = row[0:4]

        frame_count += 1
        elapsed_time = time.time() - start_time
        cv2.putText(img, f'FPS: {frame_count / elapsed_time:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255),
                    2)
        # print(f'FPS: {frame_count / elapsed_time:.2f}')
        cv2.imshow('win1', img)
        cv2.moveWindow('win1', 700, 200)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

main()
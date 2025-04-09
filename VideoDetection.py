import cv2 as cv
from Detection import detect_red_ball, centroid

cap = cv.VideoCapture("rgb_ball_720.mp4")
#frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
#frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
#fps = cap.get(cv.CAP_PROP_FPS)

#output = cv.VideoWriter("output_with_centroid.mp4", cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    processed_img, mask1 = detect_red_ball(frame)
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    final_frame = centroid(mask1, processed_img)
    #output.write(final_frame)
    #cv.imshow('Red Ball Detection - Press q to quit', final_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
#output.release()
cv.destroyAllWindows()
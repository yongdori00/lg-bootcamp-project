
import numpy as np
file = './data/eye_tracker_train_and_val.npz'
file2 = './data/fine-tuning.npz'
npzfile = np.load(file)
npzfile2 = np.load(file2)
val_eye_left = npzfile["val_eye_left"][90:150]
val_eye_right = npzfile["val_eye_right"][90:150]
val_face = npzfile["val_face"][90:150]
val_face_mask = npzfile["val_face_mask"][90:150]
val_y = npzfile["val_y"][90:150]

val_eye_left2 = npzfile2["train_eye_left"][:30:2]
val_eye_right2 = npzfile2["train_eye_right"][:30:2]
val_face2 = npzfile2["train_face"][:30:2]
val_face_mask2 = npzfile2["train_face_mask"][:30:2]

np.savez('./data/eye_tracker_train_and_val_lite.npz', 
         val_eye_left=val_eye_left,
         val_eye_right=val_eye_right,
         val_face=val_face,
         val_face_mask=val_face_mask,
         val_y=val_y,
         val_eye_left2=val_eye_left2,
         val_eye_right2=val_eye_right2,
         val_face2=val_face2,
         val_face_mask2=val_face_mask2
         )
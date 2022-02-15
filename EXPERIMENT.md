# h36m

    Namespace(actions='*', architecture='3,3,3,3,3', batch_size=1024, bone_length_term=True, by_subject=False, causal=False, channels=1024, checkpoint='checkpoint', checkpoint_frequency=10, data_augmentation=True, dataset='h36m', dense=False, disable_optimizations=False, downsample=1, dropout=0.25, epochs=60, evaluate='epoch_80.bin', export_training_curves=False, keypoints='gt', learning_rate=0.001, linear_projection=False, lr_decay=0.95, no_eval=False, no_proj=False, render=False, resume='', stride=1, subjects_test='S9,S11', subjects_train='S1,S5,S6,S7,S8', subjects_unlabeled='', subset=1, test_time_augmentation=True, viz_action=None, viz_bitrate=3000, viz_camera=0, viz_downsample=1, viz_export=None, viz_limit=-1, viz_no_ground_truth=False, viz_output=None, viz_size=5, viz_skip=0, viz_subject=None, viz_video=None, warmup=1)
    Loading dataset...
    Preparing data...
    Loading 2D detections...
    INFO: Receptive field: 243 frames
    INFO: Trainable parameter count: 16903171
    Loading checkpoint checkpoint\epoch_80.bin
    This model was trained for 80 epochs
    INFO: Testing on 2103096 frames
    Evaluating...
    ----Directions----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 47.43131893447663 mm
    Protocol #2 Error (P-MPJPE): 19.33394155793476 mm
    Protocol #3 Error (N-MPJPE): 25.207331250283207 mm
    Velocity Error (MPJVE): 23.248913475780252 mm
    ----------
    ----Discussion----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 63.48089943477081 mm
    Protocol #2 Error (P-MPJPE): 21.57475582288291 mm
    Protocol #3 Error (N-MPJPE): 31.043127913766206 mm
    Velocity Error (MPJVE): 26.4286556620478 mm
    ----------
    ----Eating----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 47.45253575738085 mm
    Protocol #2 Error (P-MPJPE): 17.391318174236005 mm
    Protocol #3 Error (N-MPJPE): 23.812264691339685 mm
    Velocity Error (MPJVE): 21.014358271688817 mm
    ----------
    ----Greeting----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 51.91472705983443 mm
    Protocol #2 Error (P-MPJPE): 22.156947209293058 mm
    Protocol #3 Error (N-MPJPE): 30.31773424498104 mm
    Velocity Error (MPJVE): 26.89125821529969 mm
    ----------
    ----Phoning----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 65.1003768052728 mm
    Protocol #2 Error (P-MPJPE): 21.444525437373425 mm
    Protocol #3 Error (N-MPJPE): 29.672409023087614 mm
    Velocity Error (MPJVE): 27.082875418054652 mm
    ----------
    ----Photo----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 65.02313487928296 mm
    Protocol #2 Error (P-MPJPE): 23.04408349989207 mm
    Protocol #3 Error (N-MPJPE): 36.398270984889706 mm
    Velocity Error (MPJVE): 28.46629338417601 mm
    ----------
    ----Posing----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 55.11991067493353 mm
    Protocol #2 Error (P-MPJPE): 21.600696354682892 mm
    Protocol #3 Error (N-MPJPE): 29.76151724914048 mm
    Velocity Error (MPJVE): 27.95794083985464 mm
    ----------
    ----Purchases----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 49.076571513940145 mm
    Protocol #2 Error (P-MPJPE): 20.126746475257747 mm
    Protocol #3 Error (N-MPJPE): 27.398168813288144 mm
    Velocity Error (MPJVE): 25.57889097202632 mm
    ----------
    ----Sitting----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 61.88941032061979 mm
    Protocol #2 Error (P-MPJPE): 22.865061313759252 mm
    Protocol #3 Error (N-MPJPE): 31.054505125697926 mm
    Velocity Error (MPJVE): 28.905978962346214 mm
    ----------
    ----SittingDown----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 93.45031925968084 mm
    Protocol #2 Error (P-MPJPE): 28.389248648675263 mm
    Protocol #3 Error (N-MPJPE): 40.87059836410109 mm
    Velocity Error (MPJVE): 35.062127113548996 mm
    ----------
    ----Smoking----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 59.929710430450456 mm
    Protocol #2 Error (P-MPJPE): 21.298797675751445 mm
    Protocol #3 Error (N-MPJPE): 28.320242999808695 mm
    Velocity Error (MPJVE): 25.852201353284322 mm
    ----------
    ----Waiting----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 51.768362755709894 mm
    Protocol #2 Error (P-MPJPE): 20.016457800993965 mm
    Protocol #3 Error (N-MPJPE): 29.10783968042056 mm
    Velocity Error (MPJVE): 25.375551553196196 mm
    ----------
    ----WalkDog----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 60.36320188540272 mm
    Protocol #2 Error (P-MPJPE): 23.43260121555544 mm
    Protocol #3 Error (N-MPJPE): 33.524719774133736 mm
    Velocity Error (MPJVE): 28.655929222687487 mm
    ----------
    ----Walking----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 44.34143078076457 mm
    Protocol #2 Error (P-MPJPE): 17.301402825007784 mm
    Protocol #3 Error (N-MPJPE): 24.049539853885904 mm
    Velocity Error (MPJVE): 20.834753922443 mm
    ----------
    ----WalkTogether----
    Test time augmentation: True
    Protocol #1 Error (MPJPE): 43.78615127206613 mm
    Protocol #2 Error (P-MPJPE): 16.985289671311847 mm
    Protocol #3 Error (N-MPJPE): 23.39951928485128 mm
    Velocity Error (MPJVE): 20.608146234092228 mm
    ----------
    Protocol #1   (MPJPE) action-wise average: 57.3 mm
    Protocol #2 (P-MPJPE) action-wise average: 21.1 mm
    Protocol #3 (N-MPJPE) action-wise average: 29.6 mm
    Velocity      (MPJVE) action-wise average: 26.13 mm

# sh36m
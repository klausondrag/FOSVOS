# https://gist.github.com/tedmiston/6060034
"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2


def show_webcam(net, mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        img = apply_network(img, net)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

def apply_network(img, net):
    return img

def main(use_resnet=True):
    if use_resnet:
        net = OSVOS_RESNET(pretrained=False)
        net.load_state_dict(torch.load('resnet18.pth', map_location=lambda storage, loc: storage))
    else:
        pass

    net = net.cuda()
    show_webcam(None, mirror=True)


if __name__ == '__main__':
    main()

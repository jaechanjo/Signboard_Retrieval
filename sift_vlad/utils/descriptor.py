import cv2
import numpy as np

# RootSIFT
class RootSIFT:
	def __init__(self, max_kps):
		# initialize the SIFT feature extractor
		self.max_kps = max_kps
		self.extractor = cv2.SIFT_create(self.max_kps)
	def compute(self, image, kps, eps=1e-7):
		# compute SIFT descriptors
		kps, descs = self.extractor.detectAndCompute(image,kps)
		# if there are no keypoints or descriptors, return an empty tuple
		if len(kps) == 0:
			return ([], None)
		# apply the Hellinger kernel by first L1-normalizing and taking the
		# square-root(L2-noramlized)
		descs /= (descs.sum(axis=1, keepdims=True) + eps)
		descs = np.sqrt(descs)
		#descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
# return a tuple of the keypoints and descriptors
		return (kps, descs)


def rootsift_extractor(img):
    '''
    Description: extract \emph{sift} feature from given image
    Input: file_path - image path
    Output: des - a list of descriptors of all the keypoint from the image
    '''
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rs = RootSIFT(99999) #full
    _, des = rs.compute(gray, None) 

    return des

#get descriptor
def get_des_vector(image_list, DESDIM=128):
    '''
    Description: get descriptors of all the images 
    Input: file_path_list - all images path
           DESDIM - SIFT local descriptor dimension 128
    Output:       all_des - a np array of all descriptors
            image_des_len - a list of number of the keypoints for each image 
    '''
    
    # all_des = np.empty(shape=[0, DESDIM]) #float64
    all_des = np.float32([]).reshape(0,DESDIM) #float32
    image_des_len = []

    for img in image_list:
        try:
            des = rootsift_extractor(img) # RootSIFT             
            all_des = np.concatenate([all_des, des]) #모든 이미지의 des vector를 concat 해서 출력! #np.concat : 다차원 배열의 결과
            image_des_len.append(len(des)) #각각 이미지의 len(des)을 따로 list에 모아두기.
        except:
            print("extract feature error")
    
    # feature num count
    featCnt = all_des.shape[0] #kp 개수
    print(str(featCnt) + " features in " + " images")
    
    return all_des, image_des_len, featCnt
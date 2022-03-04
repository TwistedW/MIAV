import numpy as np
import cv2


def calWeight(d, k):
    '''
    :param d: 融合重叠部分直径
    :param k: 融合计算权重参数
    :return:
    '''

    x = np.arange(-d / 2, d / 2)
    y = 1 / (1 + np.exp(-k * x))
    return y


def imgFusion(img1, img2, overlap, left_right=True):
    '''
    图像加权融合
    :param img1:
    :param img2:
    :param overlap: 重合长度
    :param left_right: 是否是左右融合
    :return:
    '''
    # 这里先暂时考虑平行向融合
    # w = calWeight(overlap, 0.05)  # k=5 这里是超参
    w = calWeight(overlap, 0.5)  # k=5 这里是超参

    # if left_right:  # 左右融合
    #     col, row = img1.shape
    #     img_new = np.zeros((row, 2 * col - overlap))
    #     img_new[:, :col] = img1
    #     w_expand = np.tile(w, (col, 1))  # 权重扩增
    #     img_new[:, col - overlap:col] = (1 - w_expand) * img1[:, col - overlap:col] + w_expand * img2[:, :overlap]
    #     img_new[:, col:] = img2[:, overlap:]
    if left_right:  # 左右融合
        col, row1 = img1.shape
        col, row2 = img2.shape
        img_new = np.zeros((col, row1 + row2 - overlap))
        img_new[:, :row1] = img1
        w_expand = np.tile(w, (col, 1))  # 权重扩增
        img_new[:, row1 - overlap:row1] = (1 - w_expand) * img1[:, row1 - overlap:row1] + w_expand * img2[:, :overlap]
        img_new[:, row1:] = img2[:, overlap:]
    else:  # 上下融合
        row1, col = img1.shape
        row2, col = img2.shape
        img_new = np.zeros((row1 + row2 - overlap, col))
        img_new[:row1, :] = img1
        w = np.reshape(w, (overlap, 1))
        w_expand = np.tile(w, (1, col))
        img_new[row1 - overlap:row1, :] = (1 - w_expand) * img1[row1 - overlap:row1, :] + w_expand * img2[:overlap, :]
        img_new[row1:, :] = img2[overlap:, :]
    # else:  # 上下融合
    #     row, col = img1.shape
    #     img_new = np.zeros((2 * row - overlap, col))
    #     img_new[:row, :] = img1
    #     w = np.reshape(w, (overlap, 1))
    #     w_expand = np.tile(w, (1, col))
    #     img_new[row - overlap:row, :] = (1 - w_expand) * img1[row - overlap:row, :] + w_expand * img2[:overlap, :]
    #     img_new[row:, :] = img2[overlap:, :]
    return img_new


class Stitcher:

    # 拼接函数
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # 获取输入图片
        (imageB, imageA) = images
        # 检测A、B图片的SIFT关键特征点，并计算特征描述子
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        if M is None:
            return None

        # 否则，提取匹配结果
        # H是3x3视角变换矩阵
        (matches, H, status) = M
        # 将图片A进行视角变换，result是变换后图片
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        # 将图片B传入result图片最左端
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # 检测是否需要显示图片匹配
        if showMatches:
            # 生成匹配图片
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回结果
            return (result, vis)

        # 返回匹配结果
        return result

    def detectAndDescribe(self, image):
        # 将彩色图片转换成灰度图
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        descriptor = cv2.xfeatures2d.SIFT_create()

        # 检测SIFT特征点，并计算描述子
        (kps, features) = descriptor.detectAndCompute(image, None)

        # 将结果转换成NumPy数组
        kps = np.float32([kp.pt for kp in kps])

        # 返回特征点集，及对应的描述特征
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # 建立暴力匹配器
        matcher = cv2.DescriptorMatcher_create("BruteForce")

        # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m in rawMatches:
            # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # 存储两个点在featuresA, featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 当筛选后的匹配对大于4时，计算视角变换矩阵
        if len(matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算视角变换矩阵
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # 返回结果
            return (matches, H, status)

        # 如果匹配对小于4时，返回None
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis

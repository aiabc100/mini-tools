#python编程，实现"压字印章抠图"功能
import cv2
import numpy as np

def extract_seal(image_path, output_path="seal_result.png"):
    """
    从图像中提取红色印章并保存为透明背景的PNG。
    :param image_path: 输入图像路径
    :param output_path: 输出图像路径
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像")
        return

    # 转换为HSV颜色空间（更易分离颜色）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围（红色在HSV中跨0°和180°，需两个区间）
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # 创建红色掩码
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 形态学操作去噪（可选）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 填充空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 去除小点

    # 创建透明背景结果（4通道：BGRA）
    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask  # Alpha通道设为掩码

    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"印章已提取并保存至 {output_path}")

    # 显示结果（可选）
    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Extracted Seal", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 使用示例
    extract_seal("input.jpg")  # 替换为你的图像路径
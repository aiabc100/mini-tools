import cv2
import numpy as np

def extract_all_black(input_path="sign.jpg", output_path="all_black_extracted.png"):
    """
    从图片中提取所有黑色像素，生成透明背景的PNG。
    :param input_path: 输入图片路径（JPG格式）
    :param output_path: 输出PNG图片路径
    """
    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        print("无法读取图像")
        return
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 方法1：基于阈值的黑色提取（最常用）
    # 设置黑色阈值（0-255，值越小越黑）
    black_threshold = 80  # 可调整，通常50-100之间
    _, black_mask = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 方法2：使用颜色范围提取黑色（在BGR空间中）
    # 这可以更好地处理彩色图片中的黑色
    bgr_lower = np.array([0, 0, 0])
    bgr_upper = np.array([80, 80, 80])  # 上限，可调
    black_mask_bgr = cv2.inRange(img, bgr_lower, bgr_upper)
    
    # 选择一种掩码（这里使用方法1的结果）
    black_mask = black_mask
    
    # 去除噪点
    kernel = np.ones((3, 3), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    
    # 创建透明背景（4通道）
    b, g, r = cv2.split(img)
    alpha = black_mask.copy()
    
    # 可选：平滑边缘
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0.5)
    
    # 合并通道
    result = cv2.merge([b, g, r, alpha])
    
    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"所有黑色像素已提取并保存至 {output_path}")
    
    # 显示结果
    cv2.imshow("Original", img)
    cv2.imshow("Black Mask", black_mask)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 使用示例
    extract_all_black()
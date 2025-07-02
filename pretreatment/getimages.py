# 优化版XJTU-DBUP数据采集代码（启动延迟显著降低）
# 此代码是XJTU-DBUP原始数据采集的代码（按键触发版本）
import cv2
import os
import time
from datetime import datetime



class CameraController:
    def __init__(self, cam1_index=2, cam2_index=1, resolution=(1920, 1080)):
        # 初始化摄像头
        self.cap_nc = cv2.VideoCapture(cam1_index)
        self.cap_ir = cv2.VideoCapture(cam2_index)

        # 设置分辨率
        self.resolution = resolution
        self.cap_nc.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap_nc.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap_ir.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap_ir.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # 自动参数配置
        self._configure_camera(self.cap_nc)
        self._configure_camera(self.cap_ir)

        # 验证摄像头
        if not self.cap_nc.isOpened() or not self.cap_ir.isOpened():
            raise Exception("无法访问摄像头")

    def _configure_camera(self, cap):
        """摄像头自动参数配置"""
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 自动对焦
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # 自动白平衡
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 自动曝光

    def release(self):
        """释放摄像头资源"""
        self.cap_nc.release()
        self.cap_ir.release()


def capture_session(controller, fps, total_seconds, gender, hand, main_dir):
    """
    执行单次采集任务
    """
    total_frames = fps * total_seconds

    # 创建带时间戳的文件夹
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    session_dir = os.path.join(main_dir, f"{timestamp}_{gender}_{hand}")
    nc_dir = os.path.join(session_dir, "NC")
    ir_dir = os.path.join(session_dir, "IR")
    os.makedirs(nc_dir, exist_ok=True)
    os.makedirs(ir_dir, exist_ok=True)

    print(f"\n🎬 开始采集！开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️ 采集参数: {fps}FPS, 总时长{total_seconds}秒, 总帧数{total_frames}")

    try:
        # 清空摄像头缓冲区
        for _ in range(5):
            controller.cap_nc.read()
            controller.cap_ir.read()

        start_t = time.time()
        for i in range(total_frames):
            # 采集帧
            ret_nc, frame_nc = controller.cap_nc.read()
            ret_ir, frame_ir = controller.cap_ir.read()

            if not ret_nc or not ret_ir:
                print("警告：部分帧捕获失败")
                continue

            # 保存图像（最高质量）
            frame_num = f"{i + 1:03d}"
            cv2.imwrite(os.path.join(nc_dir, f"{timestamp}_NC_{frame_num}.jpg"),
                        frame_nc, [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(os.path.join(ir_dir, f"{timestamp}_IR_{frame_num}.jpg"),
                        frame_ir, [cv2.IMWRITE_JPEG_QUALITY, 100])

            # 显示进度
            if (i + 1) % fps == 0:
                print(f"📸 已采集 {i + 1}/{total_frames} 帧 (第{(i + 1) // fps}秒)")

            # 精确帧率控制
            elapsed = time.time() - start_t
            sleep_time = (i + 1) / fps - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("⚠️ 用户中断采集！")
    finally:
        duration = time.time() - start_t
        print(f"\n✅ 采集完成！实际耗时：{duration:.2f}秒")
        print(f"📁 数据保存至：{session_dir}")


def main():
    # 配置参数
    main_dir = r"C:\Users\lifan\Desktop\DBUPOriginal0406"
    os.makedirs(main_dir, exist_ok=True)

    # 采集参数
    fps = 6
    total_seconds = 25
    gender = 'M'
    hand = 'R'

    # 初始化摄像头控制器
    try:
        controller = CameraController(
            cam1_index=0,
            cam2_index=1,
            resolution=(1920, 1080)
        )
    except Exception as e:
        print(f"初始化失败：{e}")
        return

    # 创建预览窗口
    cv2.namedWindow("实时预览", cv2.WINDOW_NORMAL)
    print("硬件准备就绪，按 S 开始采集，按 Q 退出...")

    collecting = False  # 采集状态标志
    while True:
        # 显示实时预览（仅显示NC摄像头）
        ret, frame = controller.cap_nc.read()
        if ret:
            cv2.imshow("实时预览", frame)

        # 按键检测
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or key == ord('S'):
            if not collecting:
                collecting = True
                # 开始新采集
                capture_session(controller, fps, total_seconds, gender, hand, main_dir)
                collecting = False
        elif key == ord('q') or key == ord('Q'):
            break

    # 释放资源
    controller.release()
    cv2.destroyAllWindows()
    print("程序已安全退出")


if __name__ == "__main__":
    main()
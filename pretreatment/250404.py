# 此代码是XJTU-DBUP原始数据采集的代码
import cv2
import os
import time
from datetime import datetime


def capture_high_quality_images(fps=12, total_seconds=10, cam1_index=0, cam2_index=1,
                                resolution=(3840, 2160), gender='M', hand='R'):
    """
    高画质双摄像头采集函数

    参数:
    fps: 每秒帧数 (默认12)
    total_seconds: 总采集秒数 (默认10)
    cam1_index: 摄像头1索引 (默认0)
    cam2_index: 摄像头2索引 (默认1)
    resolution: 分辨率 (宽度,高度) (默认3840x2160)
    gender: 性别标识 (M-男性, F-女性)
    hand: 左右手标识 (L-左手, R-右手)
    """
    total_frames = fps * total_seconds

    # 创建主文件夹
    main_dir = r"C:\Users\lifan\Desktop\DBUPOriginal"
    os.makedirs(main_dir, exist_ok=True)

    # 创建带性别和左右手标识的时间戳文件夹
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d%H%M%S")
    session_dir = os.path.join(main_dir, f"{timestamp}_{gender}_{hand}")
    nc_dir = os.path.join(session_dir, "NC")
    ir_dir = os.path.join(session_dir, "IR")

    os.makedirs(nc_dir, exist_ok=True)
    os.makedirs(ir_dir, exist_ok=True)

    # 初始化摄像头
    cap_nc = cv2.VideoCapture(cam1_index)
    cap_ir = cv2.VideoCapture(cam2_index)

    # 原代码设置分辨率的部分
    cap_nc.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap_nc.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap_ir.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap_ir.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # +++ 新增自动参数配置部分 +++
    def configure_camera(cap):
        # 返回配置结果字典（用于调试）
        config = {}

        # 自动对焦
        config['autofocus'] = cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        # 自动白平衡
        config['auto_wb'] = cap.set(cv2.CAP_PROP_AUTO_WB, 1)

        # 自动曝光（可能需要尝试不同的值）
        config['auto_exposure'] = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

        # 其他自动设置
        # config['brightness'] = cap.set(cv2.CAP_PROP_BRIGHTNESS, -1)  # 自动亮度

        return config

    # 配置并打印结果
    print("\n摄像头配置状态:")
    print("摄像头1 (NC):", configure_camera(cap_nc))
    print("摄像头2 (IR):", configure_camera(cap_ir))
    # +++ 结束新增部分 +++

    # 验证摄像头
    if not cap_nc.isOpened() or not cap_ir.isOpened():
        print("错误：无法访问摄像头")
        return

    print(f"🎬 开始采集！开始时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"⚙️ 采集参数: {fps}FPS, 总时长{total_seconds}秒, 总帧数{total_frames}, 分辨率{resolution[0]}x{resolution[1]}")
    print(f"🧑 受试者信息: 性别{'男' if gender == 'M' else '女'}, {'左' if hand == 'L' else '右'}手")

    try:
        start_t = time.time()
        for i in range(total_frames):
            ret_nc, frame_nc = cap_nc.read()
            ret_ir, frame_ir = cap_ir.read()

            if not ret_nc or not ret_ir:
                print("警告：部分帧捕获失败")
                continue

            # 保存图像（最高JPEG质量）
            frame_num = f"{i + 1:03d}"
            cv2.imwrite(os.path.join(nc_dir, f"{timestamp}_NC_{frame_num}.jpg"),
                        frame_nc, [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(os.path.join(ir_dir, f"{timestamp}_IR_{frame_num}.jpg"),
                        frame_ir, [cv2.IMWRITE_JPEG_QUALITY, 100])

            # 显示进度
            if (i + 1) % fps == 0:
                print(f"📸 已采集 {i + 1}/{total_frames} 帧 (第{(i + 1) // fps}秒)")

            # 精确计时
            elapsed = time.time() - start_t
            target_time = (i + 1) / fps
            sleep_time = target_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        cap_nc.release()
        cap_ir.release()
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n✅ 采集完成！结束时间：{end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱ 实际耗时：{duration.total_seconds():.2f}秒")
        print(f"📁 数据保存至：{session_dir}")


if __name__ == "__main__":
    # 在这里修改采集参数
    capture_high_quality_images(
        fps=6,  # 帧率 (帧/秒)
        total_seconds=25,  # 总采集时间 (秒)
        cam1_index=2,  # 摄像头1索引
        cam2_index=1,  # 摄像头2索引
        resolution=(1920, 1080),  # 分辨率
        gender='M',  # 性别标识: 'M'-男性, 'F'-女性
        hand='R'  # 左右手标识: 'L'-左手, 'R'-右手
    )

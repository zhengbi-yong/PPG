import matplotlib.pyplot as plt
import os


def read_and_visualize_data(file_path):
    # 初始化数据存储
    timestamps = []
    channel1 = []
    channel2 = []
    channel3 = []
    channel4 = []

    try:
        # 读取数据文件
        print(f"Attempting to open the file: {file_path}")
        with open(file_path, "r") as file:
            for line_number, line in enumerate(file, start=1):
                parts = line.strip().split()

                # 打印读取的行内容和分割结果
                print(f"Line {line_number}: {line.strip()}")
                print(f"Split parts: {parts}")

                if len(parts) == 6:  # 修改为 6 部分以适配实际数据格式
                    timestamps.append(parts[0] + " " + parts[1])  # 日期和时间
                    channel1.append(float(parts[2]))
                    channel2.append(float(parts[3]))
                    channel3.append(float(parts[4]))
                    channel4.append(float(parts[5]))
                else:
                    print(
                        f"Warning: Line {line_number} does not have 6 parts. Skipping..."
                    )

        # 检查数据是否被正确读取
        print("Data Read Summary:")
        print(f"Number of timestamps: {len(timestamps)}")
        print(f"Channel 1 values: {channel1[:5]} (showing first 5)")
        print(f"Channel 2 values: {channel2[:5]} (showing first 5)")
        print(f"Channel 3 values: {channel3[:5]} (showing first 5)")
        print(f"Channel 4 values: {channel4[:5]} (showing first 5)")

        if (
            not timestamps
            or not channel1
            or not channel2
            or not channel3
            or not channel4
        ):
            print(
                "Error: One or more channels have no data. Please check your input file format."
            )
            return

        # 创建保存图像的文件夹
        output_folder = "./PPG_BIT_txt"
        os.makedirs(output_folder, exist_ok=True)

        # 绘制并保存每个通道的图像
        channels = [channel1, channel2, channel3, channel4]
        channel_labels = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
        line_styles = ["-", "--", "-.", ":"]
        markers = ["o", "s", "^", "x"]

        for i, (channel, label, linestyle, marker) in enumerate(
            zip(channels, channel_labels, line_styles, markers), start=1
        ):
            plt.figure(figsize=(10, 6))
            plt.plot(
                timestamps, channel, label=label, marker=marker, linestyle=linestyle
            )
            plt.xlabel("Timestamp")
            plt.ylabel("Value")
            plt.title(f"{label} Data Visualization")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()

            # 保存图像
            output_path = os.path.join(output_folder, f"{label.replace(' ', '_')}.png")
            plt.savefig(output_path)
            print(f"Saved {label} plot to {output_path}")
            plt.close()  # 关闭图像以释放内存

    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except ValueError as e:
        print(f"Error: Failed to convert data to float. {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


# 设置你的数据文件路径
file_path = "D:\\PPG\\data\\PPG_BIT\\平原数据\\心率_1.txt"  # 替换为你的txt文件路径
read_and_visualize_data(file_path)

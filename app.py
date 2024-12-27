import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import trajectorytools as tt
import os
import subprocess
import platform
import torch
from io import BytesIO



python_exe = os.path.join(os.getcwd(), "_internal", "python.exe")
idtrackerai_exe = os.path.join(os.getcwd(), "_internal", "Scripts", "idtrackerai.exe")


# 初始化 session_state 中的 'page' 变量
if 'page' not in st.session_state:
    st.session_state.page = '数据分析与绘图'

# 侧边栏导航
st.sidebar.title("导航")

if st.sidebar.button("数据分析与绘图"):
    st.session_state.page = "数据分析与绘图"

# Data Analysis and Plotting Page
if st.session_state.page == "数据分析与绘图":
    # st.title("智能AI生物水质监测系统V1.0")
    st.markdown(
        "<h1 style='text-align: center;'>数据分析与绘图</h1>",
        unsafe_allow_html=True
    )
    # if st.button("启动追踪程序"):
    #     system_type = platform.system()
    #     st.info("正在分析中...")
    #     try:
    #         if system_type == "Windows":
    #             subprocess.run(['cmd', '/c', 'idtrackerai'], check=True)
    #         elif system_type in ["Linux", "Darwin"]:
    #             subprocess.run(['idtrackerai'], check=True)
    #         else:
    #             st.error("不支持的操作系统。请手动运行 idtrackerai。")
    #     except subprocess.CalledProcessError as e:
    #         st.error("Finished, Please upload trajectories csv file")
    #     except FileNotFoundError:
    #         st.error("未找到 idtrackerai，请确保已安装并配置在系统路径中。")

    # 创建一个按钮，用户点击按钮后开始视频分割
    if st.button("开始视频追踪"):
        # 自动检测系统类型并在当前终端中运行 idtrackerai 命令
        system_type = platform.system()
        st.info("正在训练卷积神经网络追踪轨迹...")
        try:
            if system_type == "Windows":
                #subprocess.run(['cmd', '/c', 'idtrackerai'], check=True)
                subprocess.run(['cmd', '/c', f'{python_exe} {idtrackerai_exe}'], check=True)
            elif system_type in ["Linux", "Darwin"]:  # Darwin 表示 macOS
                subprocess.run(['idtrackerai'], check=True)
            else:
                st.error("不支持的操作系统。请手动运行 idtrackerai。")
        except subprocess.CalledProcessError as e:
            st.error("结束，请上传轨迹csv文件")
        except FileNotFoundError:
            st.error("未找到 idtrackerai，请确保已安装并配置在系统路径中。")





    st.title("请上传轨迹文件")
    uploaded_file = st.file_uploader("上传轨迹CSV文件", type=["csv"], key="file_uploader_1")
    if uploaded_file is None:
        st.stop()
    
    df_raw = pd.read_csv(uploaded_file)
    df_raw = df_raw.fillna(method='ffill').fillna(method='bfill')
    columns = df_raw.columns
    processed_data = []
    num_individuals = len([col for col in columns if col.startswith('x')])
    
    for i in range(1, num_individuals + 1):
        x_col = f'x{i}'
        y_col = f'y{i}'
        if x_col in columns and y_col in columns:
            temp_df = pd.DataFrame({
                'frame': df_raw.index,
                'id': i,
                'pos_x': df_raw[x_col],
                'pos_y': df_raw[y_col]
            })
            processed_data.append(temp_df)
    
    df = pd.concat(processed_data, ignore_index=True)
    
    def calculate_velocity_and_acceleration(df):
        try:
            df['delta_x'] = df.groupby('id')['pos_x'].diff()
            df['delta_y'] = df.groupby('id')['pos_y'].diff()
            df['delta_frame'] = df.groupby('id')['frame'].diff()
            df['delta_frame'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df['delta_frame'].fillna(1, inplace=True)
            df['delta_x'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df['delta_y'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df['delta_x'].fillna(0, inplace=True)
            df['delta_y'].fillna(0, inplace=True)
            df['delta_x'] = pd.to_numeric(df['delta_x'], errors='coerce').fillna(0)
            df['delta_y'] = pd.to_numeric(df['delta_y'], errors='coerce').fillna(0)
            df['delta_frame'] = pd.to_numeric(df['delta_frame'], errors='coerce').fillna(1)
            df['velocity'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2) / df['delta_frame']
            df['acceleration'] = df.groupby('id')['velocity'].diff() / df['delta_frame']
            return df
        except Exception as e:
            print("Error:", e)
            return df
    
    df = calculate_velocity_and_acceleration(df)
    
    
    custom_colours = [
        (168/255, 3/255, 38/255),
        (218/255, 56/255, 42/255),
        (246/255, 121/255, 72/255),
        (253/255, 185/255, 107/255),
        (202/255, 232/255, 242/255),
        (146/255, 197/255, 222/255),
        (92/255, 144/255, 194/255),
        (57/255, 81/255, 162/255)
    ]
    
    def get_colours(num_individuals):
        return [custom_colours[i % len(custom_colours)] for i in range(num_individuals)]
    
    colours = get_colours(num_individuals)
    
    def save_figure_as_image(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=600)
        buf.seek(0)
        return buf
    
    def plot_velocity_acceleration(df, colours):
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        ids = df['id'].unique()
        for idx, id_val in enumerate(ids):
            individual_df = df[df['id'] == id_val]
            axes[0].plot(individual_df['frame'], individual_df['velocity'], label=f'ID {id_val}', color=colours[idx])
            axes[1].plot(individual_df['frame'], individual_df['acceleration'], label=f'ID {id_val}', color=colours[idx])
        axes[0].set_xlabel('frame')
        axes[0].set_ylabel('velocity')
        axes[0].legend()
        axes[1].set_xlabel('frame')
        axes[1].set_ylabel('acceleration')
        axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig)
        if st.button("绘制高分辨率速度与加速度图"):
            buf = save_figure_as_image(fig)
            st.download_button("下载速度与加速度图", buf, "velocity_acceleration.png", "image/png")
        plt.close(fig)
    
    def plot_average_velocity_acceleration(df):
        avg_velocity = df.groupby('frame')['velocity'].mean()
        avg_acceleration = df.groupby('frame')['acceleration'].mean()
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        axes[0].plot(avg_velocity.index, avg_velocity.values, color=(57/255, 81/255, 162/255))
        axes[0].set_xlabel('frame')
        axes[0].set_ylabel('average velocity')
        axes[1].plot(avg_acceleration.index, avg_acceleration.values, color=(57/255, 81/255, 162/255))
        axes[1].set_xlabel('frame')
        axes[1].set_ylabel('average acceleration')
        plt.tight_layout()
        st.pyplot(fig)
        if st.button("绘制高分辨率平均速度与加速度图"):
            buf = save_figure_as_image(fig)
            st.download_button("下载平均速度与加速度图", buf, "average_velocity_acceleration.png", "image/png")
        plt.close(fig)
    
    def plot_trajectories_old(df, colours, show_points):
        fig, ax = plt.subplots(figsize=(10, 10))
        if show_points:
            scatter_color = (168/255, 3/255, 38/255)
            ax.scatter(df['pos_x'], df['pos_y'], s=10, c=[scatter_color], alpha=0.6)
        else:
            ids = df['id'].unique()
            for idx, id_val in enumerate(ids):
                individual_df = df[df['id'] == id_val]
                ax.plot(individual_df['pos_x'], individual_df['pos_y'], label=f'ID {id_val}', color=colours[idx])
            ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        st.pyplot(fig)
        if st.button("绘制高分辨率轨迹图"):
            buf = save_figure_as_image(fig)
            st.download_button("下载轨迹图", buf, "trajectories.png", "image/png")
        plt.close(fig)


    def plot_trajectories(df, colours):
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter_color = (57/255, 81/255, 162/255)#(218/255, 56/255, 42/255)
        ax.scatter(df['pos_x'], df['pos_y'], s=10, c=[scatter_color], alpha=0.8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        st.pyplot(fig)
        if st.button("绘制高分辨率轨迹图"):
            buf = save_figure_as_image(fig)
            st.download_button("下载轨迹图", buf, "trajectories.png", "image/png")
        plt.close(fig)
    
    def plot_hexbin_heatmap_2d_dynamic(df, threshold):
        fig, ax = plt.subplots(figsize=(10, 10))
        if len(df) > 0:
            hb = ax.hexbin(df['pos_x'], df['pos_y'], gridsize=50, cmap='Blues', mincnt=1)
            heatmap_values = hb.get_array()
            heatmap_values[heatmap_values > threshold] = np.nan
            hb.set_array(heatmap_values)
            valid_values = heatmap_values[~np.isnan(heatmap_values)]
            if len(valid_values) > 0:
                hb.set_clim(0, valid_values.max())
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.tight_layout()
        st.pyplot(fig)
        if st.button("绘制高分辨率Heatmap图"):
            buf = save_figure_as_image(fig)
            st.download_button("下载Heatmap图", buf, "heatmap.png", "image/png")
        plt.close(fig)


    st.title("轨迹可视化")
    # Ensure toggle state is stored in session_state
    # if "toggle" not in st.session_state:
    #     st.session_state["toggle"] = False
    
    # # Replace checkbox with toggle switch
    # show_points = st.checkbox("修复轨迹", value=st.session_state.toggle)
    # st.session_state["toggle"] = show_points  # Update session state
    
    # plot_trajectories(df, colours, show_points)
    plot_trajectories(df, colours)
    st.title("运动Heatmap")
    max_density_threshold = st.slider(
        "Heatmap过滤阈值",
        min_value=0.0,
        max_value=1000.0,
        value=300.0,
        step=1.0
    )
    
    # Plot dynamic heatmap with threshold
    plot_hexbin_heatmap_2d_dynamic(df, max_density_threshold)

    
    st.title("速度与加速度")
    st.subheader("异常值修复")
    max_velocity = st.slider("速度阈值", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
    max_acceleration = st.slider("加速度阈值", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
    
    apply_filter = st.button("开始修复")
    if apply_filter:
        # Filter data based on thresholds
        st.session_state["filtered_df"] = df[(df['velocity'] <= max_velocity) & (df['acceleration'].abs() <= max_acceleration)]
        st.success("修复完成")
    else:
        # If button not pressed, use original data
        if "filtered_df" not in st.session_state:
            st.session_state["filtered_df"] = df
    
    df_filtered = st.session_state["filtered_df"]
    
    # Plot velocity and acceleration
    # plot_velocity_acceleration(df_filtered, colours)
    
    # Plot average velocity and acceleration
    plot_average_velocity_acceleration(df_filtered)

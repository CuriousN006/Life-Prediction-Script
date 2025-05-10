import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

def calculate_life(strain_range, T_m=35, f=48):
    """
    Engelmaier Modified Coffin-Manson Life Prediction Model
    
    Parameters:
    -----------
    strain_range : float
        Equivalent strain range from hysteresis loop
    T_m : float
        Mean temperature in °C (default: 35)
    f : float
        Frequency in cycles/day (default: 48)
    
    Returns:
    --------
    N_f : float
        Number of cycles to failure
    """
    epsilon_f = 0.323608  # 63Sn37Pb 재료의 피로 연성 계수
    
    # 피로 연성 지수 계산
    c = -0.442 - 6e-4 * T_m + 1.74e-2 * math.log(1 + f)
    
    # 전단 변형률 범위 계산
    gamma_range = math.sqrt(3) * strain_range
    
    # 수명 계산
    N_f = 0.5 * ((gamma_range / (2 * epsilon_f)) ** (1 / c))
    
    return N_f

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermomechanical Fatigue Analysis Tool")
        self.root.geometry("1000x800")
        
        # 스트레스 및 스트레인 데이터
        self.stress_data = None
        self.strain_data = None
        
        # 상단 프레임 (파일 입력용)
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10)
        
        tk.Label(top_frame, text="1. 스트레스 데이터 파일 선택:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        tk.Button(top_frame, text="파일 선택", command=self.load_stress_data).grid(row=0, column=1, padx=5, pady=5)
        self.stress_file_label = tk.Label(top_frame, text="파일이 선택되지 않았습니다.")
        self.stress_file_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')
        
        tk.Label(top_frame, text="2. 스트레인 데이터 파일 선택:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        tk.Button(top_frame, text="파일 선택", command=self.load_strain_data).grid(row=1, column=1, padx=5, pady=5)
        self.strain_file_label = tk.Label(top_frame, text="파일이 선택되지 않았습니다.")
        self.strain_file_label.grid(row=1, column=2, padx=5, pady=5, sticky='w')
        
        # 매개변수 프레임
        param_frame = tk.Frame(root)
        param_frame.pack(pady=10)
        
        tk.Label(param_frame, text="평균 온도 (T_m, °C):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.t_m_var = tk.StringVar(value="35")
        tk.Entry(param_frame, textvariable=self.t_m_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(param_frame, text="주파수 (f, cycles/day):").grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.f_var = tk.StringVar(value="48")
        tk.Entry(param_frame, textvariable=self.f_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        tk.Label(param_frame, text="피로 연성 계수 (ϵ'f):").grid(row=0, column=4, padx=5, pady=5, sticky='w')
        self.epsilon_f_var = tk.StringVar(value="0.323608")
        tk.Entry(param_frame, textvariable=self.epsilon_f_var, width=10).grid(row=0, column=5, padx=5, pady=5)
        
        # 계산 버튼
        tk.Button(param_frame, text="분석 실행", command=self.analyze, bg='lightblue', padx=10).grid(row=0, column=6, padx=15, pady=5)
        
        # 결과 프레임
        result_frame = tk.Frame(root)
        result_frame.pack(pady=10)
        
        self.result_text = tk.Text(result_frame, width=80, height=10)
        self.result_text.pack(pady=10)
        
        # 그래프 프레임
        self.graph_frame = tk.Frame(root)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def load_stress_data(self):
        file_path = filedialog.askopenfilename(title="스트레스 데이터 파일 선택",
                                              filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            try:
                self.stress_data = pd.read_csv(file_path, delimiter='\t')
                self.stress_file_label.config(text=f"로드됨: {file_path.split('/')[-1]}")
            except Exception as e:
                messagebox.showerror("Error", f"파일 로드 중 오류 발생: {str(e)}")
    
    def load_strain_data(self):
        file_path = filedialog.askopenfilename(title="스트레인 데이터 파일 선택",
                                              filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            try:
                self.strain_data = pd.read_csv(file_path, delimiter='\t')
                self.strain_file_label.config(text=f"로드됨: {file_path.split('/')[-1]}")
            except Exception as e:
                messagebox.showerror("Error", f"파일 로드 중 오류 발생: {str(e)}")
    
    def analyze(self):
        if self.stress_data is None or self.strain_data is None:
            messagebox.showerror("Error", "스트레스와 스트레인 데이터를 모두 로드해주세요.")
            return
        
        try:
            # 데이터 정리
            stress = self.stress_data.iloc[:, 2].values
            strain = self.strain_data.iloc[:, 2].values
            time = self.stress_data.iloc[:, 1].values
            
            # 세 번째 사이클만 추출 (3600초에서 5400초까지)
            third_cycle_mask = (self.stress_data.iloc[:, 1] >= 3600) & (self.stress_data.iloc[:, 1] <= 5400)
            stress_third_cycle = self.stress_data.iloc[:, 2].values[third_cycle_mask]
            strain_third_cycle = self.strain_data.iloc[:, 2].values[third_cycle_mask]

            # 세 번째 사이클의 변형률 범위 계산
            strain_range = max(strain_third_cycle) - min(strain_third_cycle)
            
            # 수명 계산
            t_m = float(self.t_m_var.get())
            f = float(self.f_var.get())
            n_f = calculate_life(strain_range, t_m, f)
            
            # 피로 연성 지수 c 계산 (표시용)
            c = -0.442 - 6e-4 * t_m + 1.74e-2 * math.log(1 + f)
            
            # 결과 표시
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"분석 결과:\n")
            self.result_text.insert(tk.END, f"==============================================\n")
            self.result_text.insert(tk.END, f"변형률 범위 (Δε): {strain_range:.6f}\n")
            self.result_text.insert(tk.END, f"전단 변형률 범위 (Δγ): {math.sqrt(3) * strain_range:.6f}\n")
            self.result_text.insert(tk.END, f"피로 연성 지수 (c): {c:.6f}\n")
            self.result_text.insert(tk.END, f"예상 수명 (Nf): {n_f:.2f} 사이클\n")
            self.result_text.insert(tk.END, f"==============================================\n")
            
            # 히스테리시스 그래프 그리기
            self.plot_hysteresis(strain, stress)
            
        except Exception as e:
            messagebox.showerror("Error", f"분석 중 오류 발생: {str(e)}")
    
    def plot_hysteresis(self, strain, stress):
        # 기존 그래프 제거
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        # 새 그래프 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # 히스테리시스 그래프
        ax1.plot(strain, stress, 'b-', marker='o', markersize=4)
        ax1.set_xlabel('Strain (mm/mm)')
        ax1.set_ylabel('Stress (MPa)')
        ax1.set_title('Stress-Strain Hysteresis Loop')
        ax1.grid(True)
        
        # 마지막 사이클 찾기 (데이터가 여러 사이클을 포함한다고 가정)
        # 간단히 하기 위해 데이터의 마지막 1/3을 마지막 사이클로 간주
        n = len(strain)
        last_third_idx = int(n * 2/3)
        
        # 시간에 따른 응력과 변형률 그래프
        time = np.arange(len(strain))
        ax2.plot(time, stress, 'r-', label='Stress (MPa)')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(time, strain, 'b-', label='Strain')
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Stress (MPa)', color='r')
        ax2_twin.set_ylabel('Strain (mm/mm)', color='b')
        ax2.set_title('Stress and Strain vs. Time')
        ax2.grid(True)
        
        # 범례 추가
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        # Tkinter에 그래프 표시
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
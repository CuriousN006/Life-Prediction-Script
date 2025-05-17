import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import traceback # 상세 에러 출력용
import platform # 현재 운영체제를 확인하기 위해 추가


# 한글 폰트 설정
try:
    font_name = None
    system_name = platform.system()

    if system_name == 'Windows':
        font_name = 'Malgun Gothic'
    elif system_name == 'Darwin': # macOS
        font_name = 'AppleGothic'
    elif system_name == 'Linux':
        # Linux에서는 NanumGothic을 많이 사용합니다.
        # fc-list :lang=ko 명령으로 사용 가능한 한글 폰트 목록을 확인할 수 있습니다.
        try:
            import matplotlib.font_manager as fm
            # 설치된 폰트 목록에서 'NanumGothic' 포함 여부 확인
            if any('NanumGothic' in f.name for f in fm.fontManager.ttflist):
                font_name = 'NanumGothic'
            else:
                # print("DEBUG: NanumGothic 폰트를 찾지 못했습니다. 다른 한글 폰트를 확인하거나 설치해주세요.")
                pass
        except ImportError:
            # print("DEBUG: matplotlib.font_manager를 임포트할 수 없습니다.")
            pass # font_manager를 사용할 수 없는 매우 제한적인 환경

    if font_name:
        plt.rcParams['font.family'] = font_name # plt.rcParams 사용
        # 한글 폰트 사용 시 마이너스 부호 깨짐 방지
        plt.rcParams['axes.unicode_minus'] = False # plt.rcParams 사용
        # print(f"DEBUG: 한글 폰트 '{font_name}'으로 설정 완료.")
    else:
        # 적절한 한글 폰트를 찾지 못한 경우라도 마이너스 부호는 처리
        plt.rcParams['axes.unicode_minus'] = False # plt.rcParams 사용
        if system_name != 'Windows' and system_name != 'Darwin': # Windows/macOS 외 시스템에서 font_name이 None이면 경고
             print(f"경고: '{system_name}' 시스템에서 한글 지원 글꼴을 자동으로 설정하지 못했습니다. 그래프에 한글이 깨질 수 있습니다. 'Nanum' 계열 글꼴 등의 설치를 권장합니다.")

except Exception as e:
    print(f"한글 폰트 설정 중 오류 발생: {e}")
    # 오류 발생 시에도 마이너스 부호 처리 시도
    try:
        plt.rcParams['axes.unicode_minus'] = False # plt.rcParams 사용
    except Exception as e_rc:
        print(f"axes.unicode_minus 설정 중 추가 오류: {e_rc}")


# calculate_life 함수 수정: epsilon_f_prime 매개변수 추가
def calculate_life(strain_range, epsilon_f_prime, T_m=35, f=48):
    """
    Engelmaier Modified Coffin-Manson Life Prediction Model
    
    Parameters:
    -----------
    strain_range : float
        Equivalent strain range from hysteresis loop
    epsilon_f_prime : float  <--- NEW
        Fatigue ductility coefficient
    T_m : float
        Mean temperature in °C (default: 35)
    f : float
        Frequency in cycles/day (default: 48)
    
    Returns:
    --------
    N_f : float
        Number of cycles to failure
    """
    # epsilon_f = 0.323608  # 하드코딩된 값 제거
    
    c = -0.442 - 6e-4 * T_m + 1.74e-2 * math.log(1 + f)
    gamma_range = math.sqrt(3) * strain_range
    
    # 0으로 나누거나, 밑이 0 또는 음수일 때 지수에 따른 오류 방지
    if epsilon_f_prime == 0: # epsilon_f_prime이 0이면 계산 불가
        return float('nan') # 또는 float('inf') 등 상황에 맞는 값
    if c == 0: # c가 0이면 1/c 에서 오류 발생
        return float('nan') 

    base = gamma_range / (2 * epsilon_f_prime) # 전달받은 epsilon_f_prime 사용
    exponent = 1 / c
    
    if gamma_range == 0: # 변형률 범위가 0이면 base는 0
        if exponent > 0: # 0의 양수 제곱은 0
            return float('inf') # 수명을 무한대로 간주 (0.5 * 0 이므로 Nf는 0이 되어야 하나, 공학적으로 수명 무한)
                                # 혹은 매우 큰 값으로 처리하거나 사용자 정의에 따름
                                # 여기서는 0.5 * (0)^exponent 이므로 Nf가 0이 될수도 있으나, 일반적으로 Nf는 사이클 수.
                                # 변형률이 0이면 파괴되지 않는다고 가정하고 inf 반환.
        elif exponent < 0: # 0의 음수 제곱은 무한대
            return 0 # 0.5 * 무한대의 역수는 0에 가까워짐 (1/무한대 -> 0)
        else: # 0의 0 제곱은 1
             return 0.5 * (1) # Nf = 0.5

    if base < 0 and exponent % 1 != 0: # 밑이 음수이고 지수가 정수가 아닌 분수일 때 복소수 발생
        return float('nan') 
    
    try:
        N_f = 0.5 * (base ** exponent)
    except (ValueError, ZeroDivisionError, OverflowError):
        N_f = float('nan')
    return N_f

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermomechanical Fatigue Analysis Tool")
        # GUI 크기는 이전과 동일하게 유지하거나 필요시 조절
        self.root.geometry("1000x880")


        self.time_data_full = None
        self.stress_values_loaded = None
        self.strain_values_loaded = None

        top_frame = tk.Frame(root)
        top_frame.pack(pady=5)

        tk.Label(top_frame, text="1. 스트레스 데이터 파일 선택:").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        tk.Button(top_frame, text="파일 선택", command=self.load_stress_data).grid(row=0, column=1, padx=5, pady=2)
        self.stress_file_label = tk.Label(top_frame, text="파일이 선택되지 않았습니다.", width=50, anchor='w')
        self.stress_file_label.grid(row=0, column=2, padx=5, pady=2, sticky='w')

        tk.Label(top_frame, text="2. 스트레인 데이터 파일 선택:").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        tk.Button(top_frame, text="파일 선택", command=self.load_strain_data).grid(row=1, column=1, padx=5, pady=2)
        self.strain_file_label = tk.Label(top_frame, text="파일이 선택되지 않았습니다.", width=50, anchor='w')
        self.strain_file_label.grid(row=1, column=2, padx=5, pady=2, sticky='w')
        
        settings_frame = tk.Frame(root)
        settings_frame.pack(pady=5, fill='x', padx=10)

        param_frame = tk.LabelFrame(settings_frame, text="수명 예측 모델 파라미터")
        param_frame.pack(side=tk.LEFT, padx=5, pady=5, fill='x', expand=True)

        tk.Label(param_frame, text="평균 온도 (T_m, °C):").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        self.t_m_var = tk.StringVar(value="35")
        tk.Entry(param_frame, textvariable=self.t_m_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        tk.Label(param_frame, text="주파수 (f, cycles/day):").grid(row=0, column=2, padx=5, pady=2, sticky='w')
        self.f_var = tk.StringVar(value="48")
        tk.Entry(param_frame, textvariable=self.f_var, width=10).grid(row=0, column=3, padx=5, pady=2)
        
        # epsilon_f_prime 입력 필드 추가
        tk.Label(param_frame, text="피로 연성 계수 (ε'f):").grid(row=0, column=4, padx=5, pady=2, sticky='w')
        self.epsilon_f_prime_var = tk.StringVar(value="0.323608") # 기본값 설정
        tk.Entry(param_frame, textvariable=self.epsilon_f_prime_var, width=12).grid(row=0, column=5, padx=5, pady=2) # 너비 약간 조절
        
        tk.Label(param_frame, text="변형률 범위 기준:").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        self.strain_source_var = tk.StringVar()
        self.strain_source_combo = ttk.Combobox(param_frame, textvariable=self.strain_source_var, state="readonly", width=25)
        self.strain_source_combo.grid(row=1, column=1, columnspan=5, padx=5, pady=2, sticky='we') # columnspan 수정

        # (cycle_def_frame 및 나머지 GUI 구성은 이전과 동일)
        cycle_def_frame = tk.LabelFrame(settings_frame, text="사이클 정의 (히스테리시스 루프용)")
        cycle_def_frame.pack(side=tk.LEFT, padx=5, pady=5, fill='x', expand=True)

        tk.Label(cycle_def_frame, text="1 사이클 시간 (초):").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        self.cycle_duration_var = tk.StringVar(value="1800")
        tk.Entry(cycle_def_frame, textvariable=self.cycle_duration_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        self.cycle_table_frame = tk.Frame(cycle_def_frame)
        self.cycle_table_frame.grid(row=1, column=0, columnspan=4, pady=5, sticky='ew')

        self.cycle_tree = ttk.Treeview(self.cycle_table_frame, columns=("cycle_no", "start_time", "end_time"), show="headings", height=4)
        self.cycle_tree.heading("cycle_no", text="사이클 No.")
        self.cycle_tree.heading("start_time", text="시작 시간 (s)")
        self.cycle_tree.heading("end_time", text="종료 시간 (s)")
        self.cycle_tree.column("cycle_no", width=80, anchor='center')
        self.cycle_tree.column("start_time", width=100, anchor='center')
        self.cycle_tree.column("end_time", width=100, anchor='center')
        
        scrollbar = ttk.Scrollbar(self.cycle_table_frame, orient="vertical", command=self.cycle_tree.yview)
        self.cycle_tree.configure(yscrollcommand=scrollbar.set)
        self.cycle_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        cycle_button_frame = tk.Frame(cycle_def_frame)
        cycle_button_frame.grid(row=2, column=0, columnspan=4, pady=2)
        tk.Button(cycle_button_frame, text="선택 사이클 플롯", command=self.add_cycle_entry_from_selection).grid(row=0, column=0, padx=5)
        tk.Button(cycle_button_frame, text="자동 사이클 추가", command=self.add_auto_cycle_entry).grid(row=0, column=1, padx=5)
        tk.Button(cycle_button_frame, text="선택 행 삭제", command=self.remove_selected_cycle_entry).grid(row=0, column=2, padx=5)
        tk.Button(cycle_button_frame, text="전체 삭제", command=self.clear_cycle_entries).grid(row=0, column=3, padx=5)

        analyze_button_frame = tk.Frame(root)
        analyze_button_frame.pack(pady=5)
        tk.Button(analyze_button_frame, text="분석 실행", command=self.analyze, bg='lightblue', padx=10, font=('Arial', 10, 'bold')).pack()

        result_frame = tk.LabelFrame(root, text="분석 결과")
        result_frame.pack(pady=5, fill='x', padx=10)
        self.result_text = tk.Text(result_frame, width=80, height=9) # 높이 약간 조절
        self.result_text.pack(pady=5, padx=5, fill='x')

        self.graph_frame = tk.Frame(root)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.update_strain_range_source_options()

    # update_strain_range_source_options, load_data_common, load_stress_data, load_strain_data는 이전과 동일
    def update_strain_range_source_options(self):
        options = ["전체 데이터 기준"]
        current_selection = self.strain_source_var.get()
        tree_items = self.cycle_tree.get_children()
        cycle_numbers_in_tree = []
        for item in tree_items:
            try:
                cycle_no_str = self.cycle_tree.item(item, "values")[0]
                cycle_numbers_in_tree.append(int(cycle_no_str))
            except (IndexError, ValueError): continue 
        for no in sorted(list(set(cycle_numbers_in_tree))):
             options.append(f"Cycle {no} 기준")
        self.strain_source_combo['values'] = options
        if current_selection in options: self.strain_source_var.set(current_selection)
        elif options: self.strain_source_var.set(options[0])
        else: self.strain_source_var.set("")

    def load_data_common(self, file_path, label_widget, data_type_name):
        try:
            data_df = pd.read_csv(file_path, delimiter='\t')
            if data_df.empty:
                raise ValueError("파일이 비어있습니다.")
            time_col_name = None
            value_col_name = None
            possible_time_cols = [col for col in data_df.columns if any(kw in col.lower() for kw in ['time', '시간'])]
            if possible_time_cols:
                time_col_name = possible_time_cols[0]
            elif data_df.shape[1] > 0 :
                time_col_name = data_df.columns[0]
                messagebox.showwarning("알림", f"{data_type_name} 파일: 명명된 시간 열('Time' 등)을 찾지 못했습니다. 첫 번째 열('{time_col_name}')을 시간으로 사용합니다.")
            else:
                raise ValueError(f"{data_type_name} 파일: 시간 데이터를 추출할 열이 없습니다.")
            found_value_col = False
            type_specific_keywords = []
            unit_keywords = []
            if data_type_name == "스트레인":
                type_specific_keywords = ['strain', '변형률']
                unit_keywords = ['mm/mm']
            elif data_type_name == "스트레스":
                type_specific_keywords = ['stress', '응력']
                unit_keywords = ['mpa']
            for col in data_df.columns:
                col_lower = col.lower()
                is_max_col = any(kw in col_lower for kw in ['maximum', '최대'])
                if is_max_col:
                    has_unit = any(unit in col_lower for unit in unit_keywords)
                    has_type = any(ts_kw in col_lower for ts_kw in type_specific_keywords)
                    if has_unit or has_type:
                        value_col_name = col
                        found_value_col = True
                        messagebox.showinfo("정보", f"{data_type_name} 파일: 단위/타입 특화 'Maximum' 열('{value_col_name}')을 값으로 사용합니다.")
                        break
            if not found_value_col:
                possible_value_cols_generic_max = [col for col in data_df.columns if any(kw in col.lower() for kw in ['maximum', '최대'])]
                if possible_value_cols_generic_max:
                    value_col_name = possible_value_cols_generic_max[0]
                    found_value_col = True
                    messagebox.showinfo("정보", f"{data_type_name} 파일: 일반 'Maximum' 열('{value_col_name}')을 값으로 사용합니다.")
            if not found_value_col:
                if data_df.shape[1] > 2: 
                    value_col_name = data_df.columns[2] 
                    messagebox.showwarning("알림", f"{data_type_name} 파일: 'Maximum' 포함 열을 찾지 못했습니다. 세 번째 열('{value_col_name}')을 값으로 사용합니다 (기존 방식).")
                else:
                    raise ValueError(f"{data_type_name} 파일: 값 데이터를 추출할 열이 부족합니다 (세 번째 열 없음).")
            time_array = data_df[time_col_name].values
            value_array = data_df[value_col_name].values
            label_widget.config(text=f"로드됨: {file_path.split('/')[-1]} (시간: '{time_col_name}', 값: '{value_col_name}')")
            return time_array, value_array
        except Exception as e:
            messagebox.showerror("오류", f"{data_type_name} 파일 로드 중 오류 발생: {str(e)}\n{traceback.format_exc()}")
            label_widget.config(text="파일 로드 실패")
            return None, None

    def load_stress_data(self):
        file_path = filedialog.askopenfilename(title="스트레스 데이터 파일 선택", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            time_arr, stress_arr = self.load_data_common(file_path, self.stress_file_label, "스트레스")
            if stress_arr is not None:
                self.stress_values_loaded = stress_arr
                if time_arr is not None and (self.time_data_full is None or len(time_arr) == len(self.stress_values_loaded)):
                    self.time_data_full = time_arr

    def load_strain_data(self):
        file_path = filedialog.askopenfilename(title="스트레인 데이터 파일 선택", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            time_arr, strain_arr = self.load_data_common(file_path, self.strain_file_label, "스트레인")
            if strain_arr is not None:
                self.strain_values_loaded = strain_arr
                if time_arr is not None:
                    if self.time_data_full is None: self.time_data_full = time_arr
                    elif len(self.time_data_full) != len(time_arr):
                         messagebox.showwarning("시간 데이터 불일치", "스트레스 파일과 스트레인 파일의 시간 데이터 포인트 수가 다릅니다. 먼저 로드된 파일의 시간 축을 기준으로 합니다.")

    def analyze(self):
        if self.stress_values_loaded is None or self.strain_values_loaded is None:
            messagebox.showerror("오류", "스트레스와 스트레인 데이터를 모두 로드해주세요.")
            return
        
        if self.time_data_full is None:
            messagebox.showerror("오류", "시간 데이터가 로드되지 않았습니다.")
            return
        
        stress_values = self.stress_values_loaded
        strain_values = self.strain_values_loaded
        time_values = self.time_data_full

        if not (len(stress_values) == len(strain_values) == len(time_values)):
            messagebox.showerror("오류", "로드된 스트레스, 스트레인, 시간 데이터의 길이가 일치하지 않습니다. 파일을 다시 확인해주세요.")
            return

        if not (strain_values.size > 0 and stress_values.size > 0 and time_values.size > 0) :
            messagebox.showerror("오류", "로드된 데이터에 유효한 값이 없습니다.")
            return
            
        try:
            # 사용자 입력 파라미터 가져오기 (epsilon_f_prime_val 추가)
            t_m = float(self.t_m_var.get())
            f = float(self.f_var.get())
            epsilon_f_prime_val = float(self.epsilon_f_prime_var.get())

            if epsilon_f_prime_val <= 0:
                messagebox.showerror("입력 오류", "피로 연성 계수 (ε'f)는 0보다 커야 합니다.")
                return

            # 변형률 범위 계산 로직 (이전과 동일)
            strain_range_source = self.strain_source_var.get()
            strain_range_for_life_calc = 0.0
            if not strain_range_source: 
                messagebox.showwarning("경고", "변형률 범위 계산 기준이 선택되지 않았습니다. '전체 데이터 기준'으로 계산합니다.")
                self.strain_source_var.set("전체 데이터 기준")
                strain_range_source = "전체 데이터 기준"

            if strain_range_source == "전체 데이터 기준":
                if strain_values.size > 0: strain_range_for_life_calc = np.max(strain_values) - np.min(strain_values)
                else: messagebox.showwarning("경고", "변형률 데이터가 없어 변형률 범위를 계산할 수 없습니다.")
            elif "Cycle" in strain_range_source:
                try:
                    selected_cycle_no_str = strain_range_source.split(' ')[1]
                    selected_cycle_no = int(selected_cycle_no_str)
                    target_cycle_def = None
                    for item in self.cycle_tree.get_children():
                        values = self.cycle_tree.item(item, "values")
                        if int(values[0]) == selected_cycle_no:
                            target_cycle_def = {"no": int(values[0]), "start_time": float(values[1]), "end_time": float(values[2])}
                            break
                    if target_cycle_def:
                        mask = (time_values >= target_cycle_def["start_time"]) & (time_values <= target_cycle_def["end_time"])
                        strain_for_calc = strain_values[mask]
                        if strain_for_calc.size > 0: strain_range_for_life_calc = np.max(strain_for_calc) - np.min(strain_for_calc)
                        else: messagebox.showwarning("경고", f"선택된 Cycle {selected_cycle_no}에 대한 변형률 데이터가 없습니다. 변형률 범위를 0으로 설정합니다.")
                    else: messagebox.showerror("오류", f"선택된 Cycle {selected_cycle_no}의 정의를 사이클 테이블에서 찾을 수 없습니다.")
                except (IndexError, ValueError) as e:
                    messagebox.showerror("오류", f"선택된 사이클 기준({strain_range_source})을 파싱하는 중 오류 발생: {e}")
            else: messagebox.showerror("오류", f"알 수 없는 변형률 범위 기준입니다: {strain_range_source}")

            # 수명 계산: epsilon_f_prime_val 전달
            n_f = calculate_life(strain_range_for_life_calc, epsilon_f_prime_val, t_m, f)
            c_val = -0.442 - 6e-4 * t_m + 1.74e-2 * math.log(1 + f)

            # 결과 표시: epsilon_f_prime_val 추가
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"분석 결과 (기준: {strain_range_source}):\n")
            self.result_text.insert(tk.END, f"==============================================\n")
            self.result_text.insert(tk.END, f"입력된 피로 연성 계수 (ε'f): {epsilon_f_prime_val:.6f}\n") # 사용된 값 표시
            self.result_text.insert(tk.END, f"계산된 변형률 범위 (Δε): {strain_range_for_life_calc:.6f}\n")
            self.result_text.insert(tk.END, f"전단 변형률 범위 (Δγ): {math.sqrt(3) * strain_range_for_life_calc:.6f}\n")
            self.result_text.insert(tk.END, f"피로 연성 지수 (c): {c_val:.6f}\n")
            self.result_text.insert(tk.END, f"예상 수명 (Nf): {n_f:.2f} 사이클\n")
            self.result_text.insert(tk.END, f"==============================================\n")

            cycle_definitions_for_plot = []
            for item in self.cycle_tree.get_children():
                values = self.cycle_tree.item(item, "values")
                try: cycle_definitions_for_plot.append({"no": int(values[0]), "start_time": float(values[1]), "end_time": float(values[2])})
                except ValueError: continue 
            self.plot_hysteresis(strain_values, stress_values, time_values, cycle_definitions_for_plot)

        except ValueError: # t_m, f, epsilon_f_prime_val 변환 실패 등
             messagebox.showerror("입력 오류", "온도, 주파수 또는 피로 연성 계수가 올바른 숫자 형식이 아닙니다.")
        except Exception as e:
            messagebox.showerror("오류", f"분석 중 오류 발생: {str(e)}\n{traceback.format_exc()}")


    # plot_hysteresis, add_cycle_entry_from_selection, add_auto_cycle_entry, remove_selected_cycle_entry, clear_cycle_entries 메소드는 이전과 동일
    def plot_hysteresis(self, strain_all, stress_all, time_all, cycle_definitions):
        # (이전과 동일한 plot_hysteresis 함수 내용)
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5)) 
        plt.subplots_adjust(bottom=0.15)

        ax1.set_xlabel('Strain (mm/mm)')
        ax1.set_ylabel('Stress (MPa)')
        ax1.set_title('Stress-Strain Hysteresis Loop (Cycles)')
        ax1.grid(True)

        colors = plt.cm.viridis(np.linspace(0, 1, len(cycle_definitions) if cycle_definitions else 1))
        if not cycle_definitions: 
            if strain_all.size >0 and stress_all.size >0 : 
                ax1.plot(strain_all, stress_all, color='gray', alpha=0.5, label="전체 데이터 (사이클 미정의)")
        else:
            for i, cycle_def in enumerate(cycle_definitions):
                mask = (time_all >= cycle_def["start_time"]) & (time_all <= cycle_def["end_time"])
                strain_cycle = strain_all[mask]
                stress_cycle = stress_all[mask]

                if strain_cycle.size > 0 and stress_cycle.size > 0:
                    ax1.plot(strain_cycle, stress_cycle, marker='o', markersize=3, linestyle='-', color=colors[i % len(colors)], label=f'Cycle {cycle_def["no"]}')
        
        if cycle_definitions or (not cycle_definitions and strain_all.size >0) : ax1.legend(fontsize='small')

        if time_all.size > 0 : 
            ax2.plot(time_all, stress_all, 'r-', label='Stress (전체)', alpha=0.3) 
            ax2_twin = ax2.twinx()
            ax2_twin.plot(time_all, strain_all, 'b-', label='Strain (전체)', alpha=0.3)
            
            if cycle_definitions:
                for i, cycle_def in enumerate(cycle_definitions):
                    mask = (time_all >= cycle_def["start_time"]) & (time_all <= cycle_def["end_time"])
                    time_cycle = time_all[mask]
                    stress_cycle = stress_all[mask]
                    strain_cycle = strain_all[mask]
                    
                    if time_cycle.size > 0:
                        line_color = colors[i % len(colors)]
                        ax2.plot(time_cycle, stress_cycle, color=line_color, linestyle='-', linewidth=1.2, label=f'Stress Cyc {cycle_def["no"]}')
                        ax2_twin.plot(time_cycle, strain_cycle, color=line_color, linestyle='--', linewidth=1.2, label=f'Strain Cyc {cycle_def["no"]}')

            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Stress (MPa)', color='r')
            ax2_twin.set_ylabel('Strain (mm/mm)', color='b')
            ax2.set_title('Stress and Strain vs. Time')
            ax2.grid(True)

            handles1, labels1 = ax2.get_legend_handles_labels()
            handles2, labels2 = ax2_twin.get_legend_handles_labels()
            
            unique_labels = {}
            combined_handles = handles1 + handles2
            combined_labels = labels1 + labels2
            temp_labels_for_unique_check = [] 

            for handle, label in zip(combined_handles, combined_labels):
                is_total_data_label = "(전체)" in label
                if is_total_data_label:
                    if label not in temp_labels_for_unique_check:
                        unique_labels[label] = handle
                        temp_labels_for_unique_check.append(label)
                else: 
                    unique_labels[label] = handle
            
            if unique_labels:
                ax2.legend(unique_labels.values(), unique_labels.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=max(1, len(unique_labels)//2), fontsize='x-small')


        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) 

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def add_cycle_entry_from_selection(self):
        if self.time_data_full is None or self.time_data_full.size == 0:
            messagebox.showwarning("경고", "데이터가 로드되지 않았거나 시간 정보가 없습니다.")
            return
        cycle_no = len(self.cycle_tree.get_children()) + 1
        start_time = 0.0
        try:
            duration = float(self.cycle_duration_var.get())
            if duration <= 0:
                messagebox.showerror("오류", "사이클 시간은 0보다 커야 합니다.")
                return
        except ValueError:
            messagebox.showerror("오류", "사이클 시간이 올바른 숫자가 아닙니다.")
            return
        end_time = duration
        if self.time_data_full is not None and self.time_data_full.size >0:
            max_data_time = self.time_data_full[-1]
            if start_time >= max_data_time:
                messagebox.showwarning("알림", "데이터의 마지막 시간 이후로는 사이클을 추가할 수 없습니다.")
                return
            end_time = min(end_time, max_data_time)
        self.cycle_tree.insert("", "end", values=(cycle_no, f"{start_time:.1f}", f"{end_time:.1f}"))
        self.update_strain_range_source_options()

    def add_auto_cycle_entry(self):
        try:
            duration = float(self.cycle_duration_var.get())
            if duration <= 0:
                messagebox.showerror("오류", "사이클 시간은 0보다 커야 합니다.")
                return
        except ValueError:
            messagebox.showerror("오류", "사이클 시간이 올바른 숫자가 아닙니다.")
            return
        children = self.cycle_tree.get_children()
        if children:
            last_item = children[-1]
            last_values = self.cycle_tree.item(last_item, "values")
            cycle_no = int(last_values[0]) + 1
            start_time = float(last_values[2]) 
        else:
            cycle_no = 1
            start_time = 0.0
        end_time = start_time + duration
        if self.time_data_full is not None and self.time_data_full.size >0:
            max_data_time = self.time_data_full[-1]
            if start_time >= max_data_time:
                messagebox.showwarning("알림", "데이터의 마지막 시간 이후로는 사이클을 추가할 수 없습니다.")
                return
            end_time = min(end_time, max_data_time)
            if start_time >= end_time and start_time >= max_data_time :
                 messagebox.showwarning("알림", "데이터의 끝에 도달했거나 유효한 사이클을 추가할 수 없습니다.")
                 return
        self.cycle_tree.insert("", "end", values=(cycle_no, f"{start_time:.1f}", f"{end_time:.1f}"))
        self.update_strain_range_source_options()

    def remove_selected_cycle_entry(self):
        selected_items = self.cycle_tree.selection()
        if not selected_items:
            messagebox.showwarning("경고", "삭제할 사이클을 선택하세요.")
            return
        for item in selected_items:
            self.cycle_tree.delete(item)
        self.update_strain_range_source_options()

    def clear_cycle_entries(self):
        for item in self.cycle_tree.get_children():
            self.cycle_tree.delete(item)
        self.update_strain_range_source_options()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
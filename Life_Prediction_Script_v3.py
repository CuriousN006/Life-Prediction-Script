import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # pyplot을 임포트하기 전에 백엔드 설정
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math

def calculate_life(strain_range, T_m=35, f=48):
    epsilon_f = 0.323608
    c = -0.442 - 6e-4 * T_m + 1.74e-2 * math.log(1 + f)
    gamma_range = math.sqrt(3) * strain_range
    if gamma_range == 0 or epsilon_f == 0 or c == 0: # 0으로 나누는 경우 또는 c가 0인 경우
        return float('inf') if c >= 0 else float('nan') # c의 부호에 따라 무한대 또는 NaN
    
    # 밑이 음수이고 지수가 분수인 경우 복소수 발생 가능성 체크
    base = gamma_range / (2 * epsilon_f)
    exponent = 1 / c
    
    if base < 0 and exponent % 1 != 0:
        # 실제 공학적 의미에 따라 처리 필요, 여기서는 NaN 반환
        # 또는 abs(base)를 사용하거나 다른 방식으로 처리
        return float('nan') 
    
    try:
        # base가 0이고 exponent가 음수면 ZeroDivisionError 발생 가능
        if base == 0 and exponent < 0:
            return float('inf') # 변형률이 0이면 수명은 무한대로 간주 (상황에 따라 다름)
        N_f = 0.5 * (base ** exponent)
    except (ValueError, ZeroDivisionError, OverflowError):
        N_f = float('nan') # 계산 오류 시 NaN 반환
    return N_f


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermomechanical Fatigue Analysis Tool")
        self.root.geometry("1000x880") # 높이 약간 늘림

        self.stress_data = None
        self.strain_data = None
        self.time_data_full = None

        # --- 상단 프레임 (파일 입력) ---
        top_frame = tk.Frame(root)
        top_frame.pack(pady=5)

        tk.Label(top_frame, text="1. 스트레스 데이터 파일 선택:").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        tk.Button(top_frame, text="파일 선택", command=self.load_stress_data).grid(row=0, column=1, padx=5, pady=2)
        self.stress_file_label = tk.Label(top_frame, text="파일이 선택되지 않았습니다.", width=40, anchor='w')
        self.stress_file_label.grid(row=0, column=2, padx=5, pady=2, sticky='w')

        tk.Label(top_frame, text="2. 스트레인 데이터 파일 선택:").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        tk.Button(top_frame, text="파일 선택", command=self.load_strain_data).grid(row=1, column=1, padx=5, pady=2)
        self.strain_file_label = tk.Label(top_frame, text="파일이 선택되지 않았습니다.", width=40, anchor='w')
        self.strain_file_label.grid(row=1, column=2, padx=5, pady=2, sticky='w')

        # --- 매개변수 및 사이클 설정 프레임 ---
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
        
        # 변형률 범위 계산 기준 선택 콤보박스
        tk.Label(param_frame, text="변형률 범위 기준:").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        self.strain_source_var = tk.StringVar()
        self.strain_source_combo = ttk.Combobox(param_frame, textvariable=self.strain_source_var, state="readonly", width=25) # 너비 조절
        self.strain_source_combo.grid(row=1, column=1, columnspan=3, padx=5, pady=2, sticky='we') # sticky 'we'로 너비 채움
        # self.update_strain_range_source_options() # 초기 옵션 설정은 __init__ 마지막으로 이동


        # --- 사이클 정의 프레임 ---
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

        # --- 분석 실행 버튼 ---
        analyze_button_frame = tk.Frame(root)
        analyze_button_frame.pack(pady=5)
        tk.Button(analyze_button_frame, text="분석 실행", command=self.analyze, bg='lightblue', padx=10, font=('Arial', 10, 'bold')).pack()

        # --- 결과 프레임 ---
        result_frame = tk.LabelFrame(root, text="분석 결과")
        result_frame.pack(pady=5, fill='x', padx=10)
        self.result_text = tk.Text(result_frame, width=80, height=8)
        self.result_text.pack(pady=5, padx=5, fill='x')

        # --- 그래프 프레임 ---
        self.graph_frame = tk.Frame(root)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.update_strain_range_source_options() # Combobox 초기 옵션 설정

    def update_strain_range_source_options(self):
        options = ["전체 데이터 기준"]
        current_selection = self.strain_source_var.get()
        
        tree_items = self.cycle_tree.get_children()
        cycle_numbers_in_tree = []
        for item in tree_items:
            try:
                cycle_no_str = self.cycle_tree.item(item, "values")[0]
                cycle_numbers_in_tree.append(int(cycle_no_str))
            except (IndexError, ValueError):
                continue 

        for no in sorted(list(set(cycle_numbers_in_tree))):
             options.append(f"Cycle {no} 기준")

        self.strain_source_combo['values'] = options
        
        if current_selection in options:
            self.strain_source_var.set(current_selection)
        elif options: 
            self.strain_source_var.set(options[0]) # 유효한 선택이 없으면 첫 번째 옵션으로
        else: 
            self.strain_source_var.set("")


    def load_data_common(self, file_path, label_widget):
        try:
            data = pd.read_csv(file_path, delimiter='\t')
            if data.shape[1] < 3:
                raise ValueError("데이터 파일은 최소 3개의 열(예: Index, Time, Value)을 가져야 합니다.")
            label_widget.config(text=f"로드됨: {file_path.split('/')[-1]}")
            
            if self.time_data_full is None or label_widget == self.stress_file_label : # 스트레스 파일 기준으로 시간 데이터 설정 (또는 둘 다 로드 시 동기화)
                 # 혹은 두 파일의 시간 열이 동일하다고 가정하고 한 번만 로드
                if data.iloc[:, 1].values.size > 0:
                    self.time_data_full = data.iloc[:, 1].values

            return data
        except Exception as e:
            messagebox.showerror("오류", f"파일 로드 중 오류 발생: {str(e)}")
            label_widget.config(text="파일 로드 실패")
            return None

    def load_stress_data(self):
        file_path = filedialog.askopenfilename(title="스트레스 데이터 파일 선택",
                                              filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            self.stress_data = self.load_data_common(file_path, self.stress_file_label)

    def load_strain_data(self):
        file_path = filedialog.askopenfilename(title="스트레인 데이터 파일 선택",
                                              filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            self.strain_data = self.load_data_common(file_path, self.strain_file_label)


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
            if start_time >= end_time and start_time >= max_data_time : # start_time == end_time 인 경우도 방지 (특히 max_data_time 도달 시)
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

    def analyze(self):
        if self.stress_data is None or self.strain_data is None:
            messagebox.showerror("오류", "스트레스와 스트레인 데이터를 모두 로드해주세요.")
            return
        
        if self.time_data_full is None:
            messagebox.showerror("오류", "시간 데이터가 로드되지 않았습니다. (데이터 파일을 다시 로드해보세요)")
            return

        try:
            stress_values = self.stress_data.iloc[:, 2].values
            strain_values = self.strain_data.iloc[:, 2].values
            time_values = self.time_data_full

            if not (strain_values.size > 0 and stress_values.size > 0 and time_values.size > 0) :
                messagebox.showerror("오류", "로드된 데이터에 유효한 값이 없습니다.")
                return

            # --- 변형률 범위 계산 ---
            strain_range_source = self.strain_source_var.get()
            strain_range_for_life_calc = 0.0

            if not strain_range_source: # 선택된 것이 없으면 (초기 상태 등)
                messagebox.showwarning("경고", "변형률 범위 계산 기준이 선택되지 않았습니다. '전체 데이터 기준'으로 계산합니다.")
                self.strain_source_var.set("전체 데이터 기준") # 기본값으로 설정
                strain_range_source = "전체 데이터 기준"


            if strain_range_source == "전체 데이터 기준":
                if strain_values.size > 0:
                    strain_range_for_life_calc = np.max(strain_values) - np.min(strain_values)
                else:
                    messagebox.showwarning("경고", "변형률 데이터가 없어 변형률 범위를 계산할 수 없습니다.")
            elif "Cycle" in strain_range_source:
                try:
                    selected_cycle_no_str = strain_range_source.split(' ')[1]
                    selected_cycle_no = int(selected_cycle_no_str)
                    
                    target_cycle_def = None
                    # 사이클 테이블에서 해당 사이클의 시간 정보 가져오기
                    for item in self.cycle_tree.get_children():
                        values = self.cycle_tree.item(item, "values")
                        if int(values[0]) == selected_cycle_no:
                            target_cycle_def = {"no": int(values[0]), "start_time": float(values[1]), "end_time": float(values[2])}
                            break
                    
                    if target_cycle_def:
                        mask = (time_values >= target_cycle_def["start_time"]) & (time_values <= target_cycle_def["end_time"])
                        strain_for_calc = strain_values[mask]
                        if strain_for_calc.size > 0:
                            strain_range_for_life_calc = np.max(strain_for_calc) - np.min(strain_for_calc)
                        else:
                            messagebox.showwarning("경고", f"선택된 Cycle {selected_cycle_no}에 대한 변형률 데이터가 없습니다. 변형률 범위를 0으로 설정합니다.")
                    else:
                        messagebox.showerror("오류", f"선택된 Cycle {selected_cycle_no}의 정의를 사이클 테이블에서 찾을 수 없습니다.")
                except (IndexError, ValueError) as e:
                    messagebox.showerror("오류", f"선택된 사이클 기준({strain_range_source})을 파싱하는 중 오류 발생: {e}")
            else: # 예상치 못한 값
                 messagebox.showerror("오류", f"알 수 없는 변형률 범위 기준입니다: {strain_range_source}")


            # --- 수명 계산 및 결과 표시 ---
            t_m = float(self.t_m_var.get())
            f = float(self.f_var.get())
            n_f = calculate_life(strain_range_for_life_calc, t_m, f)
            c_val = -0.442 - 6e-4 * t_m + 1.74e-2 * math.log(1 + f)

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"분석 결과 (기준: {strain_range_source}):\n")
            self.result_text.insert(tk.END, f"==============================================\n")
            self.result_text.insert(tk.END, f"계산된 변형률 범위 (Δε): {strain_range_for_life_calc:.6f}\n")
            self.result_text.insert(tk.END, f"전단 변형률 범위 (Δγ): {math.sqrt(3) * strain_range_for_life_calc:.6f}\n")
            self.result_text.insert(tk.END, f"피로 연성 지수 (c): {c_val:.6f}\n")
            self.result_text.insert(tk.END, f"예상 수명 (Nf): {n_f:.2f} 사이클\n")
            self.result_text.insert(tk.END, f"==============================================\n")

            # --- 그래프 플로팅 ---
            cycle_definitions_for_plot = []
            for item in self.cycle_tree.get_children():
                values = self.cycle_tree.item(item, "values")
                try:
                    cycle_definitions_for_plot.append({
                        "no": int(values[0]), "start_time": float(values[1]), "end_time": float(values[2])
                    })
                except ValueError: # 방어 코드
                    continue 
            self.plot_hysteresis(strain_values, stress_values, time_values, cycle_definitions_for_plot)

        except ValueError as ve:
             messagebox.showerror("입력 오류", f"입력 값에 문제가 있습니다: {str(ve)}")
        except Exception as e:
            messagebox.showerror("오류", f"분석 중 오류 발생: {str(e)}\n{traceback.format_exc()}") # 상세 에러 추가
            import traceback # analyze 함수 상단에 import 추가 필요할 수 있음


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
            ax1.plot(strain_all, stress_all, color='gray', alpha=0.5, label="전체 데이터 (사이클 미정의)")
        else:
            for i, cycle_def in enumerate(cycle_definitions):
                mask = (time_all >= cycle_def["start_time"]) & (time_all <= cycle_def["end_time"])
                strain_cycle = strain_all[mask]
                stress_cycle = stress_all[mask]

                if strain_cycle.size > 0 and stress_cycle.size > 0:
                    ax1.plot(strain_cycle, stress_cycle, marker='o', markersize=3, linestyle='-', color=colors[i % len(colors)], label=f'Cycle {cycle_def["no"]}')
        
        if cycle_definitions : ax1.legend(fontsize='small')

        ax2.plot(time_all, stress_all, 'r-', label='Stress (전체)', alpha=0.3) # 전체는 더 연하게
        ax2_twin = ax2.twinx()
        ax2_twin.plot(time_all, strain_all, 'b-', label='Strain (전체)', alpha=0.3) # 전체는 더 연하게
        
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
        for handle, label in zip(handles1 + handles2, labels1 + labels2):
            if label not in unique_labels: # 전체 데이터 레이블 중복 방지 및 사이클별 레이블 유지
                 if "(전체)" in label and any("(전체)" in l for l in unique_labels.keys()): # 전체 데이터는 하나만
                     if label.startswith("Stress") and not any(l.startswith("Stress (전체)") for l in unique_labels.keys()):
                         unique_labels[label] = handle
                     elif label.startswith("Strain") and not any(l.startswith("Strain (전체)") for l in unique_labels.keys()):
                         unique_labels[label] = handle
                 elif "(전체)" not in label : # 사이클별 데이터는 모두 유지
                     unique_labels[label] = handle
                 elif "(전체)" in label and not any("(전체)" in l for l in unique_labels.keys()): # 첫 전체 데이터는 추가
                     unique_labels[label] = handle


        if unique_labels:
            # 범례 항목이 너무 많아지면 그래프 가독성이 떨어지므로, loc='best' 보다는 직접 위치를 지정하거나 ncol 조절
            ax2.legend(unique_labels.values(), unique_labels.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize='x-small')


        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # 범례 공간 확보 위해 bottom 여백 조절

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
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
        try:
            import matplotlib.font_manager as fm
            if any('NanumGothic' in f.name for f in fm.fontManager.ttflist):
                font_name = 'NanumGothic'
        except ImportError:
            pass

    if font_name:
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False
    else:
        plt.rcParams['axes.unicode_minus'] = False
        if system_name not in ['Windows', 'Darwin']:
             print(f"경고: '{system_name}' 시스템에서 한글 지원 글꼴을 자동으로 설정하지 못했습니다.")

except Exception as e:
    print(f"한글 폰트 설정 중 오류 발생: {e}")
    try:
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e_rc:
        print(f"axes.unicode_minus 설정 중 추가 오류: {e_rc}")


def calculate_life(strain_range, epsilon_f_prime, T_m=35, f=48):
    """ Engelmaier Modified Coffin-Manson Life Prediction Model """
    if f <= -1: return float('nan')
    c = -0.442 - 6e-4 * T_m + 1.74e-2 * math.log(1 + f)
    gamma_range = math.sqrt(3) * strain_range
    
    if epsilon_f_prime == 0 or c == 0: return float('nan')

    base = gamma_range / (2 * epsilon_f_prime)
    exponent = 1 / c
    
    if gamma_range == 0: return float('inf')

    if base < 0 and exponent % 1 != 0: return float('nan')
    
    try:
        N_f = 0.5 * (base ** exponent)
    except (ValueError, ZeroDivisionError, OverflowError):
        N_f = float('nan')
    return N_f

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermomechanical Fatigue Analysis Tool v5")
        self.root.geometry("1000x900")

        self.time_data_full = None
        self.stress_values_loaded = None
        self.strain_values_loaded = None
        self.fig = self.ax1 = self.ax2 = None

        # GUI 구성
        top_frame = tk.Frame(root)
        top_frame.pack(pady=5, fill='x', padx=10)
        tk.Label(top_frame, text="1. 스트레스 데이터 파일 선택:").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        tk.Button(top_frame, text="파일 선택", command=self.load_stress_data).grid(row=0, column=1, padx=5, pady=2)
        self.stress_file_label = tk.Label(top_frame, text="파일이 선택되지 않았습니다.", width=60, anchor='w')
        self.stress_file_label.grid(row=0, column=2, padx=5, pady=2, sticky='w')
        tk.Label(top_frame, text="2. 스트레인 데이터 파일 선택:").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        tk.Button(top_frame, text="파일 선택", command=self.load_strain_data).grid(row=1, column=1, padx=5, pady=2)
        self.strain_file_label = tk.Label(top_frame, text="파일이 선택되지 않았습니다.", width=60, anchor='w')
        self.strain_file_label.grid(row=1, column=2, padx=5, pady=2, sticky='w')

        settings_frame = tk.Frame(root)
        settings_frame.pack(pady=5, fill='x', padx=10)

        param_frame = tk.LabelFrame(settings_frame, text="수명 예측 모델 파라미터")
        param_frame.pack(side=tk.LEFT, padx=5, pady=5, fill='x', expand=True)
        
        tk.Label(param_frame, text="최저 온도 (T_min, °C):").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        self.t_min_var = tk.StringVar(value="25")
        tk.Entry(param_frame, textvariable=self.t_min_var, width=8).grid(row=0, column=1, padx=5, pady=2)
        tk.Label(param_frame, text="최고 온도 (T_max, °C):").grid(row=0, column=2, padx=5, pady=2, sticky='w')
        self.t_max_var = tk.StringVar(value="125")
        tk.Entry(param_frame, textvariable=self.t_max_var, width=8).grid(row=0, column=3, padx=5, pady=2)
        tk.Label(param_frame, text="피로 연성 계수 (ε'f):").grid(row=0, column=4, padx=5, pady=2, sticky='w')
        self.epsilon_f_prime_var = tk.StringVar(value="0.323608")
        tk.Entry(param_frame, textvariable=self.epsilon_f_prime_var, width=12).grid(row=0, column=5, padx=5, pady=2)
        tk.Label(param_frame, text="변형률 범위 기준:").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        self.strain_source_var = tk.StringVar()
        self.strain_source_combo = ttk.Combobox(param_frame, textvariable=self.strain_source_var, state="readonly", width=25)
        self.strain_source_combo.grid(row=1, column=1, columnspan=5, padx=5, pady=2, sticky='we')

        cycle_def_frame = tk.LabelFrame(settings_frame, text="사이클 정의")
        cycle_def_frame.pack(side=tk.LEFT, padx=5, pady=5, fill='x', expand=True)
        tk.Label(cycle_def_frame, text="1 사이클 시간 (초):").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        self.cycle_duration_var = tk.StringVar(value="4200")
        self.cycle_duration_var.trace_add("write", self.update_calculated_frequency)
        tk.Entry(cycle_def_frame, textvariable=self.cycle_duration_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        tk.Label(cycle_def_frame, text="계산된 주파수:").grid(row=0, column=2, padx=5, pady=2, sticky='w')
        self.calculated_f_var = tk.StringVar()
        tk.Label(cycle_def_frame, textvariable=self.calculated_f_var, width=15).grid(row=0, column=3, padx=5, pady=2, sticky='w')
        self.cycle_table_frame = tk.Frame(cycle_def_frame)
        self.cycle_table_frame.grid(row=1, column=0, columnspan=4, pady=2, sticky='ew')
        self.cycle_tree = ttk.Treeview(self.cycle_table_frame, columns=("cycle_no", "start_time", "end_time"), show="headings", height=3)
        self.cycle_tree.heading("cycle_no", text="사이클 No."); self.cycle_tree.column("cycle_no", width=80, anchor='center')
        self.cycle_tree.heading("start_time", text="시작 시간 (s)"); self.cycle_tree.column("start_time", width=100, anchor='center')
        self.cycle_tree.heading("end_time", text="종료 시간 (s)"); self.cycle_tree.column("end_time", width=100, anchor='center')
        self.cycle_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Scrollbar(self.cycle_table_frame, orient="vertical", command=self.cycle_tree.yview).pack(side=tk.RIGHT, fill=tk.Y)
        cycle_button_frame = tk.Frame(cycle_def_frame)
        cycle_button_frame.grid(row=2, column=0, columnspan=4, pady=0)
        tk.Button(cycle_button_frame, text="선택 사이클 플롯", command=self.add_cycle_entry_from_selection).grid(row=0, column=0, padx=2)
        tk.Button(cycle_button_frame, text="자동 사이클 추가", command=self.add_auto_cycle_entry).grid(row=0, column=1, padx=2)
        tk.Button(cycle_button_frame, text="선택 행 삭제", command=self.remove_selected_cycle_entry).grid(row=0, column=2, padx=2)
        tk.Button(cycle_button_frame, text="전체 삭제", command=self.clear_cycle_entries).grid(row=0, column=3, padx=2)

        tk.Button(root, text="분석 실행", command=self.analyze, bg='lightblue', font=('Arial', 10, 'bold')).pack(pady=5)

        main_display_frame = tk.Frame(root)
        main_display_frame.pack(pady=5, fill='both', expand=True, padx=10)
        result_frame = tk.LabelFrame(main_display_frame, text="분석 결과")
        result_frame.pack(side=tk.TOP, fill='x', expand=False)
        self.result_text = tk.Text(result_frame, width=80, height=12)
        self.result_text.pack(pady=5, padx=5, fill='x')
        self.graph_frame = tk.LabelFrame(main_display_frame, text="그래프")
        self.graph_frame.pack(side=tk.BOTTOM, fill='both', expand=True, pady=(5,0))
        save_button_frame = tk.Frame(self.graph_frame)
        save_button_frame.pack(pady=2)
        self.save_btn1 = tk.Button(save_button_frame, text="응력-변형률 선도 저장", state=tk.DISABLED, command=self.save_hysteresis_plot)
        self.save_btn1.pack(side=tk.LEFT, padx=5)
        self.save_btn2 = tk.Button(save_button_frame, text="시간-데이터 그래프 저장", state=tk.DISABLED, command=self.save_time_plot)
        self.save_btn2.pack(side=tk.LEFT, padx=5)

        self.update_strain_range_source_options()
        self.update_calculated_frequency()

    def analyze(self):
        if self.stress_values_loaded is None or self.strain_values_loaded is None or self.time_data_full is None:
            messagebox.showerror("오류", "스트레스, 스트레인, 시간 데이터를 모두 로드해주세요.")
            return
        
        stress_values, strain_values, time_values = self.stress_values_loaded, self.strain_values_loaded, self.time_data_full

        if not (len(stress_values) == len(strain_values) == len(time_values)):
            messagebox.showerror("오류", "데이터 길이가 일치하지 않습니다.")
            return
        
        try:
            # 1. 공통 파라미터 준비
            t_min, t_max = float(self.t_min_var.get()), float(self.t_max_var.get())
            if t_min > t_max:
                messagebox.showerror("입력 오류", "최저 온도는 최고 온도보다 클 수 없습니다."); return
            t_m = (t_min + t_max) / 2
            
            cycle_duration_sec = float(self.cycle_duration_var.get())
            if cycle_duration_sec <= 0:
                messagebox.showerror("입력 오류", "사이클 시간은 0보다 커야 합니다."); return
            f = 86400 / cycle_duration_sec

            epsilon_f_prime_val = float(self.epsilon_f_prime_var.get())
            if epsilon_f_prime_val <= 0:
                messagebox.showerror("입력 오류", "피로 연성 계수 (ε'f)는 0보다 커야 합니다."); return

            # 2. 사용자가 선택한 기준의 수명 계산
            strain_range_source = self.strain_source_var.get()
            strain_range_for_life_calc = 0.0
            
            if strain_range_source == "전체 데이터 기준":
                if strain_values.size > 0: strain_range_for_life_calc = np.max(strain_values) - np.min(strain_values)
            elif "Cycle" in strain_range_source:
                cycle_no_str = strain_range_source.split(' ')[1]
                target_def = next((item for item in self.cycle_tree.get_children() if self.cycle_tree.item(item, "values")[0] == cycle_no_str), None)
                if target_def:
                    _, start_time, end_time = self.cycle_tree.item(target_def, "values")
                    mask = (time_values >= float(start_time)) & (time_values <= float(end_time))
                    if np.any(mask): strain_range_for_life_calc = np.max(strain_values[mask]) - np.min(strain_values[mask])
            
            n_f_selected = calculate_life(strain_range_for_life_calc, epsilon_f_prime_val, t_m, f)
            c_val_selected = -0.442 - 6e-4 * t_m + 1.74e-2 * math.log(1 + f)

            # 3. 모든 사이클에 대한 자동 분석 및 최소 수명 찾기
            min_n_f, min_life_cycle_no = float('inf'), None
            for item in self.cycle_tree.get_children():
                values = self.cycle_tree.item(item, "values")
                start_time, end_time = float(values[1]), float(values[2])
                mask = (time_values >= start_time) & (time_values <= end_time)
                if np.any(mask):
                    current_strain_range = np.max(strain_values[mask]) - np.min(strain_values[mask])
                    current_n_f = calculate_life(current_strain_range, epsilon_f_prime_val, t_m, f)
                    if not np.isnan(current_n_f) and current_n_f < min_n_f:
                        min_n_f, min_life_cycle_no = current_n_f, int(values[0])
            
            # 4. 결과 표시 (기존과 동일)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"■ 선택 기준 분석 결과 (기준: {strain_range_source})\n")
            self.result_text.insert(tk.END, "==============================================\n")
            self.result_text.insert(tk.END, f"  - 입력 온도: T_min={t_min}°C, T_max={t_max}°C (평균 T_m: {t_m:.2f}°C)\n")
            self.result_text.insert(tk.END, f"  - 입력 사이클 시간: {cycle_duration_sec}초 (주파수 f: {f:.2f} cycles/day)\n")
            self.result_text.insert(tk.END, f"  - 피로 연성 계수 (ε'f): {epsilon_f_prime_val:.6f}\n")
            self.result_text.insert(tk.END, "----------------------------------------------\n")
            self.result_text.insert(tk.END, f"  - 계산된 변형률 범위 (Δε): {strain_range_for_life_calc:.6f} mm/mm\n")
            self.result_text.insert(tk.END, f"  - 피로 연성 지수 (c): {c_val_selected:.6f}\n")
            self.result_text.insert(tk.END, f"  ▶ 예상 수명 (Nf): {n_f_selected:.2f} 사이클\n\n")
            self.result_text.insert(tk.END, "■ 최소 수명 자동 분석 결과\n")
            self.result_text.insert(tk.END, "==============================================\n")
            if min_life_cycle_no is not None:
                self.result_text.insert(tk.END, f"  ▶ 가장 짧은 예상 수명을 갖는 사이클: Cycle {min_life_cycle_no}\n")
                self.result_text.insert(tk.END, f"  ▶ 최소 예상 수명 (Nf_min): {min_n_f:.2f} 사이클\n")
            else:
                self.result_text.insert(tk.END, "  - 자동 분석을 위한 정의된 사이클이 없습니다.\n")
            self.result_text.insert(tk.END, "==============================================\n")

            # --- 5. 로직 수정: 선택된 사이클만 플롯하도록 cycle_defs_for_plot 수정 ---
            cycle_defs_for_plot = []
            if strain_range_source == "전체 데이터 기준":
                # '전체 데이터 기준'일 경우 모든 사이클을 가져옴
                cycle_defs_for_plot = [{"no": int(v[0]), "start": float(v[1]), "end": float(v[2])} for v in (self.cycle_tree.item(i, "values") for i in self.cycle_tree.get_children())]
            elif "Cycle" in strain_range_source:
                # 특정 사이클이 선택된 경우 해당 사이클만 리스트에 추가
                cycle_no_str = strain_range_source.split(' ')[1]
                target_def = next((item for item in self.cycle_tree.get_children() if self.cycle_tree.item(item, "values")[0] == cycle_no_str), None)
                if target_def:
                    v = self.cycle_tree.item(target_def, "values")
                    cycle_defs_for_plot.append({"no": int(v[0]), "start": float(v[1]), "end": float(v[2])})
            
            self.plot_hysteresis(strain_values, stress_values, time_values, cycle_defs_for_plot)

        except (ValueError, IndexError) as e:
             messagebox.showerror("입력 오류", f"입력값 처리 중 오류가 발생했습니다: {e}\n{traceback.format_exc()}")
        except Exception as e:
            messagebox.showerror("오류", f"분석 중 오류 발생: {str(e)}\n{traceback.format_exc()}")

    def plot_hysteresis(self, strain_all, stress_all, time_all, cycle_definitions):
        # 그래프 위젯 초기화
        for widget in self.graph_frame.winfo_children():
            if not isinstance(widget, tk.Button) and not isinstance(widget.master, tk.Button):
                 if widget.winfo_class() != 'Frame':
                    widget.destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
        self.fig, self.ax1, self.ax2 = fig, ax1, ax2

        plt.subplots_adjust(bottom=0.2)

        ax1.set_xlabel('Strain (mm/mm)'); ax1.set_ylabel('Stress (MPa)'); ax1.set_title('Stress-Strain Hysteresis Loop'); ax1.grid(True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(cycle_definitions) if cycle_definitions else 1))
        
        # --- 로직 수정: 전달된 cycle_definitions만 플롯 ---
        if not cycle_definitions:
            ax1.plot(strain_all, stress_all, color='gray', alpha=0.5, label="전체 데이터 (사이클 미선택)")
        else:
            for i, cycle_def in enumerate(cycle_definitions):
                mask = (time_all >= cycle_def["start"]) & (time_all <= cycle_def["end"])
                if np.any(mask):
                    ax1.plot(strain_all[mask], stress_all[mask], marker='o', markersize=2, linestyle='-', color=colors[i % len(colors)], label=f'Cycle {cycle_def["no"]}')
        
        ax1.legend(fontsize='small')
        
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Stress (MPa)', color='r'); ax2.set_title('Stress and Strain vs. Time'); ax2.grid(True)
        ax2.plot(time_all, stress_all, 'r-', label='Stress (전체)', alpha=0.2)
        ax2_twin = ax2.twinx()
        ax2_twin.set_ylabel('Strain (mm/mm)', color='b')
        ax2_twin.plot(time_all, strain_all, 'b-', label='Strain (전체)', alpha=0.2)
        
        if cycle_definitions:
            for i, cycle_def in enumerate(cycle_definitions):
                mask = (time_all >= cycle_def["start"]) & (time_all <= cycle_def["end"])
                if np.any(mask):
                    line_color = colors[i % len(colors)]
                    ax2.plot(time_all[mask], stress_all[mask], color=line_color, linewidth=1.5, label=f'Stress Cyc {cycle_def["no"]}')
                    ax2_twin.plot(time_all[mask], strain_all[mask], color=line_color, linestyle='--', linewidth=1.5, label=f'Strain Cyc {cycle_def["no"]}')
        
        handles1, labels1 = ax2.get_legend_handles_labels()
        handles2, labels2 = ax2_twin.get_legend_handles_labels()
        unique_labels = {label: handle for handle, label in reversed(list(zip(handles1 + handles2, labels1 + labels2)))}
        ax2.legend(unique_labels.values(), unique_labels.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize='x-small')

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.save_btn1.config(state=tk.NORMAL)
        self.save_btn2.config(state=tk.NORMAL)

    def _save_plot(self, target_ax, default_filename):
        if not self.fig or not target_ax:
            messagebox.showwarning("저장 불가", "저장할 그래프가 없습니다.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="그래프 저장", initialfile=default_filename, defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]
        )
        if not filepath: return
        
        renderer = self.fig.canvas.get_renderer()
        bbox = target_ax.get_tightbbox(renderer).padded(10)
        bbox_inches = bbox.transformed(self.fig.dpi_scale_trans.inverted())
        
        self.fig.savefig(filepath, bbox_inches=bbox_inches, dpi=300)
        messagebox.showinfo("저장 완료", f"그래프가 성공적으로 저장되었습니다:\n{filepath}")

    def save_hysteresis_plot(self): self._save_plot(self.ax1, "Hysteresis_Loop_Plot.png")
    def save_time_plot(self): self._save_plot(self.ax2, "Time_Series_Plot.png")
    
    def update_calculated_frequency(self, *args):
        try:
            duration_sec = float(self.cycle_duration_var.get())
            if duration_sec > 0: self.calculated_f_var.set(f"{86400 / duration_sec:.2f} cycles/day")
            else: self.calculated_f_var.set("유효하지 않음")
        except (ValueError, ZeroDivisionError): self.calculated_f_var.set("계산 불가")

    def update_strain_range_source_options(self):
        options = ["전체 데이터 기준"] + [f"Cycle {self.cycle_tree.item(i, 'values')[0]} 기준" for i in self.cycle_tree.get_children()]
        current_selection = self.strain_source_var.get()
        self.strain_source_combo['values'] = options
        if current_selection not in options: self.strain_source_var.set(options[0])

    def load_data_common(self, file_path, label_widget, data_type_name):
        try:
            data_df = pd.read_csv(file_path, delimiter='\t')
            if data_df.empty: raise ValueError("파일이 비어있습니다.")
            time_col = next((c for c in data_df.columns if 'time' in c.lower() or '시간' in c.lower()), data_df.columns[0])
            val_col = next((c for c in data_df.columns if 'max' in c.lower() or '최대' in c.lower()), data_df.columns[2] if data_df.shape[1] > 2 else None)
            if not val_col: raise ValueError("값 열을 찾을 수 없습니다.")
            
            unit_info, factor = "", 1.0
            if '[' in val_col and ']' in val_col:
                unit = val_col.split('[')[1].split(']')[0].lower()
                if data_type_name == "스트레스" and unit == 'pa': factor, unit_info = 1e-6, " (단위: Pa -> MPa 변환됨)"
                else: unit_info = f" (단위: {unit})"
            
            label_widget.config(text=f"로드됨: {file_path.split('/')[-1]}{unit_info}")
            return data_df[time_col].values, data_df[val_col].values * factor
        except Exception as e:
            messagebox.showerror("오류", f"{data_type_name} 파일 로드 중 오류: {e}"); return None, None

    def load_stress_data(self):
        path = filedialog.askopenfilename(title="스트레스 데이터 파일", filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if path:
            time, stress = self.load_data_common(path, self.stress_file_label, "스트레스")
            if stress is not None: self.stress_values_loaded, self.time_data_full = stress, time

    def load_strain_data(self):
        path = filedialog.askopenfilename(title="스트레인 데이터 파일", filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if path:
            time, strain = self.load_data_common(path, self.strain_file_label, "스트레인")
            if strain is not None:
                self.strain_values_loaded = strain
                if self.time_data_full is None: self.time_data_full = time
                elif len(self.time_data_full) != len(time): messagebox.showwarning("경고", "데이터 포인트 수가 다릅니다.")

    def add_cycle_entry(self, start_time, end_time):
        cycle_no = (int(self.cycle_tree.item(self.cycle_tree.get_children()[-1], "values")[0]) + 1) if self.cycle_tree.get_children() else 1
        max_time = self.time_data_full[-1] if self.time_data_full is not None else float('inf')
        if start_time >= max_time: messagebox.showwarning("알림", "더 이상 사이클을 추가할 수 없습니다."); return
        self.cycle_tree.insert("", "end", values=(cycle_no, f"{start_time:.1f}", f"{min(end_time, max_time):.1f}"))
        self.update_strain_range_source_options()

    def add_cycle_entry_from_selection(self):
        try:
            duration = float(self.cycle_duration_var.get())
            if duration <= 0: raise ValueError("사이클 시간은 0보다 커야 합니다.")
            self.add_cycle_entry(0.0, duration)
        except ValueError as e: messagebox.showerror("오류", str(e))

    def add_auto_cycle_entry(self):
        try:
            duration = float(self.cycle_duration_var.get())
            if duration <= 0: raise ValueError("사이클 시간은 0보다 커야 합니다.")
            start = float(self.cycle_tree.item(self.cycle_tree.get_children()[-1], "values")[2]) if self.cycle_tree.get_children() else 0.0
            self.add_cycle_entry(start, start + duration)
        except ValueError as e: messagebox.showerror("오류", str(e))

    def remove_selected_cycle_entry(self):
        items = self.cycle_tree.selection()
        if not items: messagebox.showwarning("경고", "삭제할 사이클을 선택하세요.")
        else: self.cycle_tree.delete(*items); self.update_strain_range_source_options()

    def clear_cycle_entries(self):
        self.cycle_tree.delete(*self.cycle_tree.get_children()); self.update_strain_range_source_options()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
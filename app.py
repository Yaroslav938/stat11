import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools
import warnings
import io

# Отключаем предупреждения
warnings.filterwarnings("ignore")

st.set_page_config(page_title="🔬 StatPack OmniLab v18", layout="wide", page_icon="🐠")

# ══════════════════════════════════════════════
# БЛОК 1: ДВИЖКИ ПАРСИНГА И УТИЛИТЫ ЭКСПОРТА
# ══════════════════════════════════════════════

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def convert_df_to_csv_with_index(df):
    return df.to_csv(index=True).encode('utf-8')

@st.cache_data
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Анализ_данных')
    return output.getvalue()

@st.cache_data
def smart_parse_headers(file, header_rows):
    """Универсальный парсер: склеивает многоуровневые шапки"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, header=None)
        else:
            df = pd.read_excel(file, header=None)
            
        if header_rows > 0:
            headers = df.iloc[:header_rows].ffill(axis=1)
            new_cols = []
            for col_idx in range(headers.shape[1]):
                col_vals = headers.iloc[:, col_idx].values
                clean_vals = [str(v).strip() for v in col_vals if pd.notna(v) and str(v).lower() != 'nan']
                col_name = " | ".join(clean_vals) if clean_vals else f"Столбец_{col_idx}"
                new_cols.append(col_name)
            
            df.columns = new_cols
            df = df.iloc[header_rows:].reset_index(drop=True)
            
        return df
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        return None

@st.cache_data
def parse_rnf_special(file, n_simulations=10):
    """Спец-парсер: для таблиц, где даны только min и max значения"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, header=None)
        else:
            df = pd.read_excel(file, header=None)

        min_max_row_idx = None
        for idx, row in df.iterrows():
            row_str = [str(val).lower().strip() for val in row.values]
            if 'min' in row_str and 'max' in row_str:
                min_max_row_idx = idx
                break

        if min_max_row_idx is not None:
            features_raw = df.iloc[min_max_row_idx - 1].values
            current_feature = "Unknown"
            feature_map = []
            for p in features_raw:
                p_str = str(p).strip()
                if pd.notna(p) and p_str != "" and p_str.lower() != "nan" and "зона" not in p_str.lower():
                    current_feature = p_str
                feature_map.append(current_feature)

            data_rows = df.iloc[min_max_row_idx + 1:]
            parsed_data = []

            for _, row in data_rows.iterrows():
                object_name = row.iloc[0]
                if pd.isna(object_name) or str(object_name).strip() == "":
                    continue

                for col_idx in range(1, len(row)):
                    col_type = str(df.iloc[min_max_row_idx, col_idx]).lower().strip()
                    if col_type in ['min', 'max']:
                        val = pd.to_numeric(row.iloc[col_idx], errors='coerce')
                        if pd.notna(val):
                            parsed_data.append({
                                "ID": str(object_name).strip(),
                                "Признак": feature_map[col_idx],
                                "Тип": col_type,
                                "Значение": val
                            })

            long_df = pd.DataFrame(parsed_data)
            if long_df.empty: return None, None
                
            pivot_df = long_df.pivot_table(index=['ID', 'Признак'], columns='Тип', values='Значение').reset_index()
            pivot_df['Mid'] = (pivot_df['min'] + pivot_df['max']) / 2
            
            simulated = []
            for _, row in pivot_df.dropna().iterrows():
                if row['min'] == row['max']:
                    vals = np.full(n_simulations, row['min'])
                else:
                    vals = np.random.uniform(row['min'], row['max'], n_simulations)
                for v in vals:
                    simulated.append({"ID": row['ID'], "Признак": row['Признак'], "Значение": v})
                    
            sim_df = pd.DataFrame(simulated)
            wide_df = pivot_df.pivot_table(index="ID", columns="Признак", values="Mid").fillna(0)
            return wide_df, sim_df
        return None, None
    except Exception as e:
        st.error(f"Ошибка парсинга спец-формата: {e}")
        return None, None

def parse_multi_group_files(files):
    """Спец-парсер для многофайловых сравнений во времени (например, динамика)"""
    all_long = []
    for file in files:
        fname = file.name.replace('.csv', '').replace('.xlsx', '')
        group_name = fname.split('-')[-1].strip() if '-' in fname else fname

        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, header=None)
            else:
                df = pd.read_excel(file, header=None)

            # Формат: 1 строка - даты/периоды, 2 строка - метрики
            periods = df.iloc[0].ffill().astype(str).values
            features = df.iloc[1].astype(str).values
            data = df.iloc[2:].copy()

            for col_idx in range(len(df.columns)):
                period = periods[col_idx]
                feat = features[col_idx]

                if pd.isna(feat) or feat == 'nan' or feat == '' or 'Unnamed' in feat:
                    continue

                vals = pd.to_numeric(data.iloc[:, col_idx], errors='coerce')
                valid_vals = vals.dropna()

                if not valid_vals.empty:
                    temp_df = pd.DataFrame({
                        "Группа": group_name,
                        "Период": period.replace('nan', 'Общий').strip(),
                        "Признак": feat.replace('nan', 'Метрика').strip(),
                        "Значение": valid_vals.values
                    })
                    all_long.append(temp_df)
        except Exception as e:
            st.error(f"Не удалось обработать файл {file.name}: {e}")

    if not all_long: return None, None

    full_long_df = pd.concat(all_long, ignore_index=True)
    full_long_df['ID'] = full_long_df['Группа']
    pseudo_wide = full_long_df.groupby(['ID', 'Признак'])['Значение'].mean().unstack().fillna(0)
    
    return pseudo_wide, full_long_df

# ══════════════════════════════════════════════
# БЛОК 2: МАТЕМАТИКА И СТАТИСТИКА
# ══════════════════════════════════════════════

def cohens_d(x, y):
    """Расчет размера эффекта (Cohen's d)"""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    if dof <= 0: return 0
    poolsd = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    if poolsd == 0: return 0
    return (np.mean(x) - np.mean(y)) / poolsd

def perform_pairwise_tests(df, group_col, val_col, parametric=False):
    """Post-Hoc тесты с поправкой Бонферрони и Cohen's d"""
    groups = df[group_col].unique()
    results = []
    
    for g1, g2 in itertools.combinations(groups, 2):
        d1 = df[df[group_col] == g1][val_col].dropna().values
        d2 = df[df[group_col] == g2][val_col].dropna().values
        
        if len(d1) < 2 or len(d2) < 2: continue
            
        if np.var(d1) == 0 and np.var(d2) == 0 and np.mean(d1) == np.mean(d2):
            p = 1.0
        else:
            if parametric:
                stat, p = stats.ttest_ind(d1, d2, equal_var=False)
            else:
                stat, p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
            
        effect_size = abs(cohens_d(d1, d2))
        
        if effect_size >= 0.8: eff_str = "Высокий"
        elif effect_size >= 0.5: eff_str = "Средний"
        elif effect_size >= 0.2: eff_str = "Малый"
        else: eff_str = "Незначительный"
            
        results.append((g1, g2, p, effect_size, eff_str))
        
    if not results: return pd.DataFrame()
        
    res_df = pd.DataFrame(results, columns=['Группа 1', 'Группа 2', 'p_raw', "Cohen's d", "Эффект"])
    n_tests = len(res_df)
    res_df['p_adj (Bonf)'] = (res_df['p_raw'] * n_tests).clip(upper=1.0)
    
    return res_df[['Группа 1', 'Группа 2', 'p_adj (Bonf)', "Cohen's d", "Эффект"]]

def calc_mode(series):
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan

# ══════════════════════════════════════════════
# ИНТЕРФЕЙС ЛАБОРАТОРИИ (UI)
# ══════════════════════════════════════════════

st.title("📈 StatPack OmniLab v18: Final Analytics")
st.markdown("*Универсальная станция анализа данных со встроенными алгоритмами оценки роста, выживаемости и эффективности.*")

# ── БОКОВАЯ ПАНЕЛЬ (НАСТРОЙКИ) ─────────────────────────────
with st.sidebar:
    st.header("📂 1. Загрузка данных")
    uploaded_files = st.file_uploader("Файлы Excel / CSV", type=["csv", "xlsx"], accept_multiple_files=True)
    
    st.markdown("---")
    st.header("⚙️ 2. Конфигурация парсера")
    parse_mode = st.selectbox(
        "Режим чтения данных:",
        [
            "Одиночный файл (Плоские/Объединенные шапки)", 
            "Сравнение Групп / Динамика (Несколько файлов)",
            "Генератор симуляций (только min/max)"
        ]
    )
    
    if parse_mode == "Одиночный файл (Плоские/Объединенные шапки)":
        header_rows = st.number_input("Количество строк в шапке (для склейки):", 1, 5, 1)
        n_simulations = 10
    elif parse_mode == "Генератор симуляций (только min/max)":
        n_simulations = st.slider("Точек симуляции (N)", 3, 30, 10)
        header_rows = 1
    else:
        st.info("💡 Названия групп будут взяты из названий файлов. Шапка: 1 строка - Даты/Этапы, 2 строка - Признаки.")
        header_rows = 2
        n_simulations = 10
        
    st.markdown("---")
    st.header("🎨 3. Глобальные настройки")
    alpha_level = st.selectbox("Уровень значимости (α):", [0.05, 0.01, 0.10], index=0)
    color_theme = st.selectbox("Цветовая палитра:", ["Viridis", "Plasma", "Turbo", "Spectral", "RdBu_r", "Plotly3"], index=0)

# ── ПОДГОТОВКА И ОЧИСТКА ДАННЫХ ─────────────────────────────────────────
if uploaded_files:
    wide_df, long_df = None, None
    is_ready = False

    if parse_mode == "Генератор симуляций (только min/max)":
        with st.spinner("Сборка и симуляция данных..."):
            wide_df, long_df = parse_rnf_special(uploaded_files[0], n_simulations)
            if wide_df is not None:
                st.sidebar.success(f"✅ Готово: {len(wide_df)} объектов")
                is_ready = True

    elif parse_mode == "Сравнение Групп / Динамика (Несколько файлов)":
        with st.spinner("Интеграция файлов и выравнивание временных рядов..."):
            wide_df, long_df = parse_multi_group_files(uploaded_files)
            if wide_df is not None:
                st.sidebar.success(f"✅ Готово! Групп: {len(wide_df)}, Записей: {len(long_df)}")
                is_ready = True

    elif parse_mode == "Одиночный файл (Плоские/Объединенные шапки)":
        raw_df = smart_parse_headers(uploaded_files[0], header_rows)
        if raw_df is not None:
            with st.expander("🛠 Настройка датасета (Обязательно выберите столбцы)", expanded=True):
                st.dataframe(raw_df.head(3), use_container_width=True)
                col_id, col_features = st.columns([1, 2])
                
                with col_id:
                    id_col = st.selectbox("Укажите столбец с объектами (ID, Группы):", options=raw_df.columns, index=0)
                with col_features:
                    possible_features = [c for c in raw_df.columns if c != id_col]
                    feature_cols = st.multiselect("Выберите признаки (Переменные) для анализа:", options=possible_features, default=possible_features)
                
                if id_col and feature_cols:
                    try:
                        clean_df = raw_df[[id_col] + feature_cols].copy()
                        clean_df.rename(columns={id_col: "ID"}, inplace=True)
                        for col in feature_cols:
                            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
                        
                        long_df = clean_df.melt(id_vars=["ID"], value_vars=feature_cols, var_name="Признак", value_name="Значение").dropna()
                        wide_df = clean_df.groupby("ID")[feature_cols].mean().fillna(0)
                        wide_df = wide_df.loc[:, (wide_df != 0).any(axis=0)]
                        st.sidebar.success(f"✅ Готово: {len(wide_df)} объектов")
                        is_ready = True
                    except Exception as e:
                        st.error(f"Ошибка конвертации типов: {e}")

    # ── ЛАБОРАТОРИЯ (ВКЛАДКИ) ────────────────────────
    if is_ready and wide_df is not None and not wide_df.empty:
        has_time = "Период" in long_df.columns
        
        st.markdown("---")
        t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
            "📊 1. Описательная (EDA)", 
            "📈 2. Регрессия", 
            "🌳 3. Кластеры (PCA)", 
            "🔬 4. Сравнение (ANOVA)", 
            "⚖️ 5. A/B Тестирование",
            "📑 6. Сводная Статистика",
            "🧮 7. Индексы Прироста (Био)",
            "🗄 8. Экспорт"
        ])

        # ── ВКЛАДКА 1. EDA И ДИНАМИКА ──────────────────────────────────────────
        with t1:
            st.markdown("### Визуализация распределений и профилей")
            if has_time:
                st.markdown("#### Динамика показателей во времени (Line Chart)")
                dyn_df = long_df.groupby(["Период", "Группа", "Признак"])["Значение"].agg(['mean', 'std']).reset_index()
                dyn_df['std'] = dyn_df['std'].fillna(0)
                dyn_df = dyn_df.sort_values(by="Период")
                
                dyn_feat = st.selectbox("Показатель для графика динамики:", long_df["Признак"].unique())
                fig_dyn = px.line(dyn_df[dyn_df["Признак"] == dyn_feat], x="Период", y="mean", color="Группа", 
                                  error_y="std", markers=True, title=f"Изменение: {dyn_feat} во времени",
                                  labels={"mean": "Среднее значение", "Период": "Дата / Этап"})
                st.plotly_chart(fig_dyn, use_container_width=True)
                st.markdown("---")

            c1, c2 = st.columns([1.5, 1])
            with c1:
                fig_heat = px.imshow(wide_df, color_continuous_scale=color_theme, aspect="auto", title="Тепловая матрица средних значений (сводная)")
                st.plotly_chart(fig_heat, use_container_width=True)
            with c2:
                sel_ids = st.multiselect("Объекты/Группы для Радара:", options=wide_df.index.tolist(), default=wide_df.index.tolist()[:3] if len(wide_df)>=3 else wide_df.index.tolist())
                if sel_ids and len(wide_df.columns) >= 3:
                    fig_radar = go.Figure()
                    for s_id in sel_ids:
                        vals = wide_df.loc[s_id].values.tolist()
                        fig_radar.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=wide_df.columns.tolist()+[wide_df.columns[0]], fill='toself', name=str(s_id)))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Многомерное сравнение профилей")
                    st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("---")
            col_bar, col_viol = st.columns(2)
            with col_bar:
                group_col = "Группа" if has_time else "ID"
                bar_df = long_df.groupby([group_col, "Признак"])["Значение"].agg(['mean', 'std']).reset_index()
                fig_bar = px.bar(bar_df, x=group_col, y="mean", color="Признак", barmode="group", error_y="std", title="Средние значения с планками (SD)")
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_viol:
                x_viol_col = "Период" if has_time else "Признак"
                color_viol_col = "Группа" if has_time else "ID"
                fig_violin = px.violin(long_df, x=x_viol_col, y="Значение", color=color_viol_col, box=True, points="all", title="Скрипичный график (Плотность)")
                st.plotly_chart(fig_violin, use_container_width=True)

        # ── ВКЛАДКА 2. РЕГРЕССИЯ И КОРРЕЛЯЦИЯ ───────────────────────────────
        with t2:
            st.markdown("### Корреляционный и Регрессионный анализ")
            if len(wide_df.columns) > 1:
                c1, c2 = st.columns([1, 2])
                with c1:
                    fig_corr = px.imshow(wide_df.corr(), text_auto=".2f", color_continuous_scale=color_theme, title="Критерий Пирсона (Матрица)")
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                with c2:
                    st.markdown("#### Линейная регрессия (МНК)")
                    reg_col1, reg_col2 = st.columns(2)
                    with reg_col1: x_feat = st.selectbox("Независимая (Ось X):", wide_df.columns, index=0)
                    with reg_col2: y_feat = st.selectbox("Зависимая (Ось Y):", wide_df.columns, index=1)
                    
                    if x_feat != y_feat:
                        if has_time: reg_df = long_df.pivot_table(index=["Группа", "Период", long_df.groupby(["Группа", "Период"]).cumcount()], columns="Признак", values="Значение").dropna()
                        else: reg_df = long_df.pivot_table(index=["ID", long_df.groupby("ID").cumcount()], columns="Признак", values="Значение").dropna()

                        if not reg_df.empty and x_feat in reg_df.columns and y_feat in reg_df.columns:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(reg_df[x_feat], reg_df[y_feat])
                            fig_reg = px.scatter(reg_df, x=x_feat, y=y_feat, opacity=0.6, title=f"Зависимость: {y_feat} от {x_feat}", size_max=10)
                            
                            x_range = np.linspace(reg_df[x_feat].min(), reg_df[x_feat].max(), 100)
                            fig_reg.add_trace(go.Scatter(x=x_range, y=slope * x_range + intercept, mode='lines', name='Линия OLS', line=dict(color='red')))
                            st.plotly_chart(fig_reg, use_container_width=True)
                            
                            st.markdown(f"**Уравнение:** `y = {slope:.3f}*x + {intercept:.3f}` | **R²:** {r_value**2:.3f} | **p-value:** {p_value:.4e}")
                        else:
                            st.warning("Недостаточно данных для регрессии.")
            else: st.warning("Требуется минимум 2 признака.")

        # ── ВКЛАДКА 3. PCA & TREES ──────────────────────────────────────────
        with t3:
            st.markdown("### Кластеризация и Дендрограммы")
            if len(wide_df) >= 3 and len(wide_df.columns) >= 2:
                X_scaled = StandardScaler().fit_transform(wide_df)
                
                try:
                    fig_dendro = ff.create_dendrogram(X_scaled, labels=wide_df.index.tolist(), color_threshold=2.5)
                    fig_dendro.update_layout(height=400, margin=dict(b=100)); fig_dendro.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_dendro, use_container_width=True)
                except Exception: pass

                st.markdown("#### Метод главных компонент (PCA Biplot)")
                pca_col1, pca_col2 = st.columns([1, 3])
                with pca_col1:
                    km = KMeans(n_clusters=st.slider("KMeans кластеры:", 2, min(8, len(wide_df)-1), 3), random_state=42, n_init=10)
                    pca = PCA(n_components=2)
                    pca_coords = pca.fit_transform(X_scaled)
                    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                    st.metric("PC1", f"{pca.explained_variance_ratio_[0]*100:.1f}%"); st.metric("PC2", f"{pca.explained_variance_ratio_[1]*100:.1f}%")
                    pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"], index=wide_df.index)
                    pca_df["Cluster"] = km.fit_predict(X_scaled).astype(str)

                with pca_col2:
                    fig_pca = px.scatter(pca_df.reset_index(), x="PC1", y="PC2", color="Cluster", text="ID", size_max=15, height=500)
                    fig_pca.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='black')))
                    for i, feature in enumerate(wide_df.columns):
                        fig_pca.add_annotation(x=loadings[i, 0]*3.5, y=loadings[i, 1]*3.5, ax=0, ay=0, text=feature, showarrow=True, arrowcolor="red")
                    st.plotly_chart(fig_pca, use_container_width=True)
            else: st.warning("Требуется ≥3 объектов и ≥2 признака.")

        # ── ВКЛАДКА 4. МНОЖЕСТВЕННЫЕ СРАВНЕНИЯ (ANOVA / KW) ───────────────────────────────
        with t4:
            st.markdown("### Множественные сравнения (Анализ дисперсий всех групп)")
            c_f1, c_f2 = st.columns(2)
            with c_f1: feature = st.selectbox("Целевой признак:", long_df["Признак"].unique(), key="anova_feat")
            
            df_stat = long_df[long_df["Признак"] == feature]
            if has_time:
                with c_f2: target_time = st.selectbox("Период (Время):", df_stat["Период"].unique(), key="anova_time")
                df_stat = df_stat[df_stat["Период"] == target_time]

            groups = [group["Значение"].values for name, group in df_stat.groupby("ID")]
            group_names = [name for name, group in df_stat.groupby("ID")]
            valid_groups = [g for g in groups if len(g) >= 3]
            
            if len(valid_groups) < 3: st.error("⚠️ Нужно минимум 3 объекта, имеющих ≥3 реплики.")
            else:
                c_assump1, c_assump2 = st.columns([1, 2])
                with c_assump1:
                    shapiro_p = min([stats.shapiro(g)[1] if np.var(g) > 0 else 1.0 for g in valid_groups])
                    levene_p = stats.levene(*valid_groups)[1] if any(np.var(g) > 0 for g in valid_groups) else 1.0
                    use_parametric = (shapiro_p > alpha_level) and (levene_p > alpha_level)
                    
                    st.write(f"**Нормальность:** p={shapiro_p:.4e} {'✅' if shapiro_p > alpha_level else '❌'}")
                    st.write(f"**Дисперсии:** p={levene_p:.4e} {'✅' if levene_p > alpha_level else '❌'}")
                    st.info(f"💡 Выбран: **{'ANOVA' if use_parametric else 'Kruskal-Wallis'}**")
                
                with c_assump2:
                    fig_qq = go.Figure()
                    for name, group in zip(group_names, groups):
                        if len(group) >= 3 and np.var(group) > 0:
                            osm, osr = stats.probplot(group, dist="norm")[0]
                            fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name=str(name)))
                    fig_qq.update_layout(title="QQ-График (Оценка нормальности)", height=300, margin=dict(t=30, b=10))
                    st.plotly_chart(fig_qq, use_container_width=True)

                st.markdown("---")
                if use_parametric: stat, p_omnibus = stats.f_oneway(*valid_groups)
                else: stat, p_omnibus = stats.kruskal(*valid_groups)
                st.write(f"**p-value (Общий тест):** {p_omnibus:.4e}")
                
                if p_omnibus < alpha_level:
                    st.success(f"✅ Выявлены значимые отличия. Post-Hoc анализ (Поправка Бонферрони).")
                    posthoc_df = perform_pairwise_tests(df_stat, "ID", "Значение", parametric=use_parametric)
                    ph1, ph2 = st.columns([1, 1])
                    with ph1:
                        st.dataframe(posthoc_df.style.map(lambda x: 'background-color: #a8e6cf; color: black' if isinstance(x, float) and x < alpha_level else '', subset=['p_adj (Bonf)']), use_container_width=True)
                    with ph2:
                        matrix = pd.DataFrame(index=group_names, columns=group_names, dtype=float)
                        for _, row in posthoc_df.iterrows():
                            matrix.loc[row['Группа 1'], row['Группа 2']] = row['p_adj (Bonf)']
                            matrix.loc[row['Группа 2'], row['Группа 1']] = row['p_adj (Bonf)']
                        np.fill_diagonal(matrix.values, 1.0)
                        fig_ph = px.imshow(matrix, color_continuous_scale="Reds_r", zmin=0, zmax=alpha_level, title="Матрица p-value")
                        st.plotly_chart(fig_ph, use_container_width=True)
                else: st.warning(f"⚠️ Статистически значимых отличий нет.")

        # ── ВКЛАДКА 5. A/B ТЕСТИРОВАНИЕ ─────────────────
        with t5:
            st.markdown("### A/B Тестирование (Точное сравнение двух групп)")
            c_ab1, c_ab2 = st.columns(2)
            with c_ab1: ab_feature = st.selectbox("1. Признак для A/B теста:", long_df["Признак"].unique(), key="ab_feat")
            
            df_ab_stat = long_df[long_df["Признак"] == ab_feature]
            if has_time:
                with c_ab2: target_time_ab = st.selectbox("2. Период:", df_ab_stat["Период"].unique(), key="ab_time")
                df_ab_stat = df_ab_stat[df_ab_stat["Период"] == target_time_ab]

            ab_objects = st.multiselect("3. Выберите ровно ДВЕ группы:", df_ab_stat["ID"].unique(), max_selections=2, key="ab_objs")
            
            if len(ab_objects) == 2:
                gA_name, gB_name = ab_objects[0], ab_objects[1]
                gA = df_ab_stat[df_ab_stat["ID"] == gA_name]["Значение"].dropna().values
                gB = df_ab_stat[df_ab_stat["ID"] == gB_name]["Значение"].dropna().values
                
                if len(gA) >= 3 and len(gB) >= 3:
                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.metric(f"Среднее: {gA_name}", f"{np.mean(gA):.3f}", f"n = {len(gA)}")
                    c_m2.metric(f"Среднее: {gB_name}", f"{np.mean(gB):.3f}", f"n = {len(gB)}")
                    c_m3.metric("Разница (Delta B - A)", f"{np.mean(gB) - np.mean(gA):.3f}")
                    
                    st.markdown("#### Результаты Тестов")
                    col_t1, col_t2 = st.columns(2)
                    levene_p = stats.levene(gA, gB)[1] if (np.var(gA) > 0 or np.var(gB) > 0) else 1.0
                    
                    if np.var(gA) == 0 and np.var(gB) == 0 and np.mean(gA) == np.mean(gB): t_p, mw_p = 1.0, 1.0
                    else:
                        _, t_p = stats.ttest_ind(gA, gB, equal_var=(levene_p > alpha_level))
                        _, mw_p = stats.mannwhitneyu(gA, gB, alternative='two-sided')
                    
                    with col_t1:
                        st.metric("t-критерий Стьюдента (p)", f"{t_p:.4e}")
                        if t_p < alpha_level: st.success("✅ Различаются")
                        else: st.warning("⚠️ Нет различий")
                    with col_t2:
                        st.metric("U-критерий Манна-Уитни (p)", f"{mw_p:.4e}")
                        if mw_p < alpha_level: st.success("✅ Различаются")
                        else: st.warning("⚠️ Нет различий")
                            
                    st.markdown(f"**Размер эффекта (Cohen's d):** `{abs(cohens_d(gA, gB)):.3f}`")
                    
                    fig_ab = px.histogram(df_ab_stat[df_ab_stat["ID"].isin([gA_name, gB_name])], x="Значение", color="ID", barmode="overlay", marginal="box", opacity=0.7)
                    st.plotly_chart(fig_ab, use_container_width=True)
                else: st.warning("⚠️ Нужно минимум 3 реплики в каждой группе.")

        # ── ВКЛАДКА 6. СВОДНАЯ СТАТИСТИКА ───────────────────────────────
        with t6:
            st.markdown("### Подробная сводная статистика")
            group_cols = ['ID', 'Период', 'Признак'] if has_time else ['ID', 'Признак']
            
            stats_summary_df = long_df.groupby(group_cols)['Значение'].agg(
                Количество='count', Среднее='mean', Медиана='median', Мода=calc_mode,
                Минимум='min', Максимум='max', Ст_откл='std', Дисперсия='var'
            ).reset_index()
            
            numeric_cols = ['Среднее', 'Медиана', 'Мода', 'Минимум', 'Максимум', 'Ст_откл', 'Дисперсия']
            stats_summary_df[numeric_cols] = stats_summary_df[numeric_cols].round(4)
            
            st.dataframe(stats_summary_df, use_container_width=True, height=400)
            try:
                st.download_button("📥 Скачать в Excel (.xlsx)", convert_df_to_excel(stats_summary_df), "summary_statistics.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception:
                st.download_button("📥 Скачать в CSV", convert_df_to_csv(stats_summary_df), "summary_statistics.csv", "text/csv")


        # ── ВКЛАДКА 7. КАЛЬКУЛЯТОР ПРИРОСТА И ЭФФЕКТИВНОСТИ ───────────────────────────────
        with t7:
            st.markdown("### 🧮 Автоматизированный расчет рыбоводно-биологических параметров (Рост и Индексы)")
            st.caption("Полноценная замена сложным формулам Excel. Расчет выживаемости, приростов, коэффициента Фультона и кормовой конверсии (FCR).")
            
            c_calc1, c_calc2, c_calc3 = st.columns(3)
            
            if has_time:
                with c_calc1:
                    calc_mass = st.selectbox("Признак Массы (Вес):", long_df["Признак"].unique())
                    calc_len = st.selectbox("Признак Длины (для упитанности):", ["Не учитывать"] + list(long_df["Признак"].unique()))
                    len_unit = st.radio("Единицы измерения длины:", ["мм", "см"], horizontal=True)
                with c_calc2:
                    periods_sorted = sorted(long_df["Период"].unique().tolist())
                    p_start = st.selectbox("Начальный этап:", periods_sorted, index=0)
                    p_end = st.selectbox("Конечный этап:", periods_sorted, index=len(periods_sorted)-1 if len(periods_sorted)>1 else 0)
                with c_calc3:
                    days = st.number_input("Продолжительность (дней):", min_value=1, value=30)
                    feed = st.number_input("Израсходовано корма на 1 особь за период (г):", min_value=0.0, value=0.0, step=10.0, help="Оставьте 0, если расчет FCR не нужен.")
            else:
                with c_calc1:
                    mass_start_feat = st.selectbox("Масса (Начало):", wide_df.columns)
                    mass_end_feat = st.selectbox("Масса (Конец):", wide_df.columns)
                with c_calc2:
                    len_start_feat = st.selectbox("Длина (Начало):", ["Не учитывать"] + list(wide_df.columns))
                    len_end_feat = st.selectbox("Длина (Конец):", ["Не учитывать"] + list(wide_df.columns))
                    len_unit = st.radio("Единицы длины:", ["мм", "см"], horizontal=True)
                with c_calc3:
                    days = st.number_input("Продолжительность (дней):", min_value=1, value=30)
                    feed = st.number_input("Израсходовано корма на 1 особь (г):", min_value=0.0, value=0.0, step=10.0)

            st.markdown("---")
            calc_results = []
            unique_groups = long_df["Группа"].unique() if has_time else wide_df.index
            
            for grp in unique_groups:
                try:
                    if has_time:
                        df_grp = long_df[long_df["Группа"] == grp]
                        start_data = df_grp[df_grp["Период"] == p_start]
                        end_data = df_grp[df_grp["Период"] == p_end]

                        # Количество особей для расчета выживаемости
                        N_n = start_data[start_data["Признак"] == calc_mass]["Значение"].count()
                        N_k = end_data[end_data["Признак"] == calc_mass]["Значение"].count()

                        W_n = start_data[start_data["Признак"] == calc_mass]["Значение"].mean()
                        W_k = end_data[end_data["Признак"] == calc_mass]["Значение"].mean()

                        L_n, L_k = 0, 0
                        if calc_len != "Не учитывать":
                            L_n = start_data[start_data["Признак"] == calc_len]["Значение"].mean()
                            L_k = end_data[end_data["Признак"] == calc_len]["Значение"].mean()
                    else:
                        N_n, N_k = 1, 1
                        W_n = wide_df.loc[grp, mass_start_feat]
                        W_k = wide_df.loc[grp, mass_end_feat]
                        L_n = wide_df.loc[grp, len_start_feat] if len_start_feat != "Не учитывать" else 0
                        L_k = wide_df.loc[grp, len_end_feat] if len_end_feat != "Не учитывать" else 0

                    if pd.isna(W_n) or pd.isna(W_k): continue

                    # Формулы рыбоводно-биологических параметров
                    survival = (N_k / N_n) * 100 if N_n > 0 else 0
                    A = W_k - W_n
                    C = A / days if days > 0 else 0
                    O = (A / W_n) * 100 if W_n > 0 else 0
                    Cw = ((np.log(W_k) - np.log(W_n)) / days) * 100 if W_n > 0 and W_k > 0 else 0
                    
                    # Поправка Фультона (100 для СМ, 100000 для ММ)
                    f_multiplier = 100000 if len_unit == "мм" else 100
                    K_n = (W_n / (L_n ** 3)) * f_multiplier if L_n > 0 else np.nan
                    K_k = (W_k / (L_k ** 3)) * f_multiplier if L_k > 0 else np.nan
                    
                    FCR = feed / A if (A > 0 and feed > 0) else np.nan
                    
                    # Ежесуточный рацион (% от средней массы)
                    avg_mass = (W_n + W_k) / 2
                    daily_ration = ((feed / days) / avg_mass) * 100 if (avg_mass > 0 and feed > 0) else np.nan

                    calc_results.append({
                        "Группа": grp,
                        "Выживаемость (%)": round(survival, 1) if has_time else "-",
                        "Масса начало (г)": round(W_n, 2),
                        "Масса конец (г)": round(W_k, 2),
                        "Абс. прирост (A), г": round(A, 2),
                        "Среднесут. прирост (C), г": round(C, 3),
                        "Относит. прирост (O), %": round(O, 2),
                        "Удельная скорость (SGR), %": round(Cw, 3),
                        "Упитанность начало (K)": round(K_n, 3) if pd.notna(K_n) else "-",
                        "Упитанность конец (K)": round(K_k, 3) if pd.notna(K_k) else "-",
                        "Конверсия корма (FCR)": round(FCR, 3) if pd.notna(FCR) else "-",
                        "Ежесут. рацион (%)": round(daily_ration, 3) if pd.notna(daily_ration) else "-"
                    })
                except Exception: pass

            if calc_results:
                df_growth = pd.DataFrame(calc_results)
                st.success("✅ Биологические индексы успешно рассчитаны!")
                st.dataframe(df_growth, use_container_width=True)
                
                try:
                    st.download_button("📥 Скачать таблицу расчетов в Excel", convert_df_to_excel(df_growth), "bio_parameters.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except:
                    st.download_button("📥 Скачать таблицу расчетов (CSV)", convert_df_to_csv(df_growth), "bio_parameters.csv", "text/csv")
                
                # --- НОВЫЙ БЛОК: Статистика и визуализация рассчитанных индексов ---
                st.markdown("---")
                st.markdown("#### 📊 Статистика по рассчитанным параметрам")
                
                # Подготавливаем числовой датафрейм (заменяем "-" на NaN)
                num_growth_df = df_growth.copy()
                for col in num_growth_df.columns:
                    if col != "Группа":
                        num_growth_df[col] = pd.to_numeric(num_growth_df[col].replace("-", np.nan), errors='coerce')
                        
                stat_cols = [c for c in num_growth_df.columns if c != "Группа"]
                if stat_cols:
                    growth_stats = num_growth_df[stat_cols].agg(['mean', 'median', 'min', 'max', 'std']).T
                    growth_stats.rename(columns={'mean': 'Среднее', 'median': 'Медиана', 'min': 'Минимум', 'max': 'Максимум', 'std': 'Ст. откл.'}, inplace=True)
                    st.dataframe(growth_stats.round(3), use_container_width=True)
                
                st.markdown("#### 📉 Графический анализ прироста и эффективности")
                cg1, cg2 = st.columns(2)
                
                with cg1:
                    if "Масса начало (г)" in num_growth_df.columns and "Масса конец (г)" in num_growth_df.columns:
                        df_melt = num_growth_df.melt(id_vars=["Группа"], value_vars=["Масса начало (г)", "Масса конец (г)"], var_name="Этап", value_name="Масса (г)")
                        fig_m = px.bar(df_melt, x="Группа", y="Масса (г)", color="Этап", barmode="group", 
                                       title="Изменение массы (Начало vs Конец)", color_discrete_sequence=["#636EFA", "#2CA02C"])
                        st.plotly_chart(fig_m, use_container_width=True)
                        
                with cg2:
                    if "Удельная скорость (SGR), %" in num_growth_df.columns:
                        fig_s = px.bar(num_growth_df, x="Группа", y="Удельная скорость (SGR), %", color="Группа", 
                                       title="Удельная скорость роста (SGR, %)", text_auto='.2f')
                        fig_s.update_traces(textposition='outside')
                        st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.warning("Нет данных для расчета по выбранным параметрам. Проверьте правильность выбора столбцов.")


        # ── ВКЛАДКА 8. ЭКСПОРТ ДАННЫХ ───────────────────────────────────────────
        with t8:
            st.markdown("### Хранилище агрегированных и сырых данных")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown("**Агрегированная матрица (Wide Form)**")
                st.dataframe(wide_df, use_container_width=True)
                st.download_button("💾 Скачать матрицу (CSV)", convert_df_to_csv_with_index(wide_df), "wide_data.csv", "text/csv")
                
            with col_v2:
                st.markdown("**Сырые данные (Long Form)**")
                st.dataframe(long_df, use_container_width=True)
                st.download_button("💾 Скачать сырые данные (CSV)", convert_df_to_csv(long_df), "long_data.csv", "text/csv")

else:
    st.info("👈 Загрузите ваши файлы (или файл) в панели слева для начала работы.")
    st.markdown("""
    ### 🔬 Добро пожаловать в StatPack OmniLab (v18)
    **Новое в текущей версии (Aqua & Biology Ultimate):**
    * 🧮 **Модуль биологических индексов:** Мы перенесли всю математику из ваших сложных Excel-алгоритмов прямо в приложение. 
    * ⚖️ Учтена правильная физическая шкала: **коэффициент Фультона** теперь умный. Если вы меряете рыбу в **миллиметрах**, программа сама применит коэффициент $10^5$, чтобы индекс считался точно!
    * 🐠 **Выживаемость:** Программа сама пересчитает строки в ваших файлах и выдаст точный процент выживаемости между периодами.
    * 🍽️ **Кормовые параметры:** Добавлен расчет **FCR** (кормовой конверсии) и **Ежесуточного рациона** (в % от массы тела).""")
    
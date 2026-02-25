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

st.set_page_config(page_title="🔬 StatPack OmniLab v14", layout="wide", page_icon="📈")

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
    """Экспорт DataFrame в формат Excel (.xlsx)"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Сводная_статистика')
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

# ══════════════════════════════════════════════
# БЛОК 2: МАТЕМАТИКА И СТАТИСТИКА
# ══════════════════════════════════════════════

def cohens_d(x, y):
    """Расчет размера эффекта (Cohen's d) с защитой от деления на 0"""
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
    """Вспомогательная функция для расчета моды в pandas groupby"""
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan

# ══════════════════════════════════════════════
# ИНТЕРФЕЙС ЛАБОРАТОРИИ (UI)
# ══════════════════════════════════════════════

st.title("📈 StatPack OmniLab v14: Data Science Edition")
st.markdown("*Универсальная мульти-аналитическая станция для любых типов данных (Биология, Химия, Экономика, Социология).*")

# ── БОКОВАЯ ПАНЕЛЬ (НАСТРОЙКИ) ─────────────────────────────
with st.sidebar:
    st.header("📂 1. Загрузка данных")
    uploaded_file = st.file_uploader("Файл Excel / CSV", type=["csv", "xlsx"])
    
    st.markdown("---")
    st.header("⚙️ 2. Конфигурация парсера")
    parse_mode = st.selectbox(
        "Режим чтения данных:",
        ["Универсальный (Плоские/Объединенные шапки)", "Генератор симуляций (только min/max)"],
        help="Универсальный - для любых таблиц. Генератор симуляций - для таблиц, где указаны диапазоны значений вместо сырых точек."
    )
    
    if parse_mode == "Универсальный (Плоские/Объединенные шапки)":
        header_rows = st.number_input("Количество строк в шапке (для склейки):", 1, 5, 1)
        n_simulations = 10
    else:
        n_simulations = st.slider("Точек симуляции (N)", 3, 30, 10)
        header_rows = 1
        
    st.markdown("---")
    st.header("🎨 3. Глобальные настройки")
    alpha_level = st.selectbox("Уровень значимости (α):", [0.05, 0.01, 0.10], index=0)
    color_theme = st.selectbox("Цветовая палитра:", ["Viridis", "Plasma", "Turbo", "Spectral", "RdBu_r", "Plotly3"], index=0)

# ── ПОДГОТОВКА И ОЧИСТКА ДАННЫХ ─────────────────────────────────────────
if uploaded_file:
    wide_df, long_df = None, None
    is_ready = False

    if parse_mode == "Генератор симуляций (только min/max)":
        with st.spinner("Сборка и симуляция данных..."):
            wide_df, long_df = parse_rnf_special(uploaded_file, n_simulations)
            if wide_df is not None:
                st.sidebar.success(f"✅ Загружено объектов: {len(wide_df)}")
                is_ready = True
            else:
                st.error("Ошибка формата. Попробуйте 'Универсальный' режим.")

    elif parse_mode == "Универсальный (Плоские/Объединенные шапки)":
        raw_df = smart_parse_headers(uploaded_file, header_rows)
        if raw_df is not None:
            with st.expander("🛠 Настройка датасета (Обязательно выберите столбцы)", expanded=True):
                st.dataframe(raw_df.head(3), use_container_width=True)
                col_id, col_features = st.columns([1, 2])
                
                with col_id:
                    id_col = st.selectbox("Укажите столбец с объектами (ID, Названия, Группы):", options=raw_df.columns, index=0)
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
                        wide_df = wide_df.loc[:, (wide_df != 0).any(axis=0)] # Чистка столбцов, где только нули
                        st.sidebar.success(f"✅ Готово: {len(wide_df)} объектов")
                        is_ready = True
                    except Exception as e:
                        st.error(f"Ошибка конвертации типов данных: {e}")

    # ── ЛАБОРАТОРИЯ (ВКЛАДКИ) ────────────────────────
    if is_ready and wide_df is not None and not wide_df.empty:
        st.markdown("---")
        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "📊 1. Описательная (EDA)", 
            "📈 2. Регрессия", 
            "🌳 3. Кластеры (PCA)", 
            "🔬 4. Множественные сравнения (ANOVA)", 
            "⚖️ 5. A/B Тестирование (t-тесты)",
            "📑 6. Сводная Статистика",
            "🗄 7. Экспорт"
        ])

        # ── ВКЛАДКА 1. EDA ──────────────────────────────────────────────────
        with t1:
            st.markdown("### Визуализация распределений и профилей")
            
            c1, c2 = st.columns([1.5, 1])
            with c1:
                # Heatmap
                fig_heat = px.imshow(wide_df, color_continuous_scale=color_theme, aspect="auto",
                                     title="Тепловая матрица средних значений")
                st.plotly_chart(fig_heat, use_container_width=True)
            with c2:
                # Радар
                sel_ids = st.multiselect("Выберите объекты для Радара:", options=wide_df.index.tolist(),
                                         default=wide_df.index.tolist()[:3] if len(wide_df)>=3 else wide_df.index.tolist())
                if sel_ids and len(wide_df.columns) >= 3:
                    fig_radar = go.Figure()
                    for s_id in sel_ids:
                        vals = wide_df.loc[s_id].values.tolist()
                        fig_radar.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=wide_df.columns.tolist()+[wide_df.columns[0]], fill='toself', name=str(s_id)))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Многомерное сравнение (Spider Chart)")
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("Для радара нужно ≥3 признаков.")

            st.markdown("---")
            # Столбчатые диаграммы (Bar Charts) с планками погрешностей
            st.markdown("#### Столбчатая диаграмма (Группированная с планками погрешностей SD)")
            bar_df = long_df.groupby(["ID", "Признак"])["Значение"].agg(['mean', 'std']).reset_index()
            bar_df['std'] = bar_df['std'].fillna(0) # Защита от единичных реплик
            
            fig_bar = px.bar(bar_df, x="ID", y="mean", color="Признак", barmode="group",
                             error_y="std", title="Средние значения признаков (с планками стандартного отклонения)",
                             labels={"mean": "Среднее значение", "ID": "Объект (Группа)"})
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("---")
            # Скрипичные графики
            st.markdown("#### Скрипичный график (Violin Plot) - Анализ плотности")
            st.caption("Показывает форму распределения, медиану, квартили (внутренний бокс) и все сырые точки выборки.")
            fig_violin = px.violin(long_df, x="Признак", y="Значение", color="Признак", 
                                   box=True, points="all", hover_data=["ID"],
                                   title="Распределение значений по всем признакам")
            st.plotly_chart(fig_violin, use_container_width=True)


        # ── ВКЛАДКА 2. РЕГРЕССИЯ И КОРРЕЛЯЦИЯ ───────────────────────────────
        with t2:
            st.markdown("### Корреляционный и Регрессионный анализ")
            if len(wide_df.columns) > 1:
                c1, c2 = st.columns([1, 2])
                with c1:
                    fig_corr = px.imshow(wide_df.corr(), text_auto=".2f", color_continuous_scale=color_theme,
                                         title="Критерий Пирсона (Матрица корреляций)")
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                with c2:
                    st.markdown("#### Линейная регрессия (Метод наименьших квадратов)")
                    st.caption("Оценка степени линейной зависимости между двумя любыми признаками.")
                    
                    reg_col1, reg_col2 = st.columns(2)
                    with reg_col1: x_feat = st.selectbox("Независимая переменная (Ось X):", wide_df.columns, index=0)
                    with reg_col2: y_feat = st.selectbox("Зависимая переменная (Ось Y):", wide_df.columns, index=1 if len(wide_df.columns)>1 else 0)
                    
                    if x_feat != y_feat:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(wide_df[x_feat], wide_df[y_feat])
                        r_squared = r_value**2
                        
                        fig_reg = px.scatter(wide_df.reset_index(), x=x_feat, y=y_feat, text="ID", 
                                             title=f"Зависимость: {y_feat} от {x_feat}", size_max=10)
                        fig_reg.update_traces(textposition='top center')
                        
                        x_range = np.linspace(wide_df[x_feat].min(), wide_df[x_feat].max(), 100)
                        y_range = slope * x_range + intercept
                        fig_reg.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='Линия OLS', line=dict(color='red', width=2)))
                        
                        st.plotly_chart(fig_reg, use_container_width=True)
                        
                        st.markdown(f"**Уравнение прямой:** `y = {slope:.3f} * x + {intercept:.3f}`")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("R² (Коэф. детерминации)", f"{r_squared:.3f}")
                        m2.metric("p-value (Значимость тренда)", f"{p_value:.4e}")
                        m3.metric("Стандартная ошибка", f"{std_err:.3f}")
                        
                        if p_value < alpha_level:
                            st.success(f"✅ Выявлена статистически значимая линейная зависимость (p < {alpha_level})")
                        else:
                            st.warning(f"⚠️ Линейная зависимость статистически незначима (p ≥ {alpha_level})")
                    else:
                        st.info("Пожалуйста, выберите разные признаки для X и Y.")
            else:
                st.warning("Для анализа связей требуется минимум 2 признака.")


        # ── ВКЛАДКА 3. PCA & TREES ──────────────────────────────────────────
        with t3:
            st.markdown("### Поиск скрытых паттернов и Кластеризация")
            if len(wide_df) >= 3 and len(wide_df.columns) >= 2:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(wide_df)
                
                st.markdown("#### Иерархическая кластеризация (Дендрограмма сходства)")
                try:
                    fig_dendro = ff.create_dendrogram(X_scaled, labels=wide_df.index.tolist(), color_threshold=2.5)
                    fig_dendro.update_layout(height=450, margin=dict(b=100))
                    fig_dendro.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_dendro, use_container_width=True)
                except Exception:
                    st.warning("Недостаточно математической вариативности для построения дерева.")

                st.markdown("---")
                st.markdown("#### Метод главных компонент (PCA Biplot)")
                pca_col1, pca_col2 = st.columns([1, 3])
                
                with pca_col1:
                    n_clusters = st.slider("Ожидаемое число кластеров (KMeans):", 2, min(8, len(wide_df)-1), 3)
                    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = km.fit_predict(X_scaled).astype(str)
                    
                    pca = PCA(n_components=2)
                    pca_coords = pca.fit_transform(X_scaled)
                    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                    
                    st.metric("Вклад 1-й компоненты (PC1)", f"{pca.explained_variance_ratio_[0]*100:.1f}%")
                    st.metric("Вклад 2-й компоненты (PC2)", f"{pca.explained_variance_ratio_[1]*100:.1f}%")
                    
                    pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"], index=wide_df.index)
                    pca_df["Cluster"] = clusters

                with pca_col2:
                    fig_pca = px.scatter(pca_df.reset_index(), x="PC1", y="PC2", color="Cluster", text="ID", size_max=15, height=600)
                    fig_pca.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='black')))
                    for i, feature in enumerate(wide_df.columns):
                        fig_pca.add_annotation(x=loadings[i, 0]*3.5, y=loadings[i, 1]*3.5, ax=0, ay=0, text=feature, showarrow=True, arrowhead=2, arrowcolor="red")
                    st.plotly_chart(fig_pca, use_container_width=True)
            else:
                st.warning("Для кластеризации требуется ≥3 объектов и ≥2 признака.")


        # ── ВКЛАДКА 4. МНОЖЕСТВЕННЫЕ СРАВНЕНИЯ (ANOVA / KW) ───────────────────────────────
        with t4:
            st.markdown("### Множественные сравнения (Анализ дисперсий всех групп одновременно)")
            feature = st.selectbox("Выберите целевой признак (переменную) для анализа дисперсий:", wide_df.columns, key="anova_feat")
            
            df_stat = long_df[long_df["Признак"] == feature]
            groups = [group["Значение"].values for name, group in df_stat.groupby("ID")]
            group_names = [name for name, group in df_stat.groupby("ID")]
            
            valid_groups = [g for g in groups if len(g) >= 3]
            
            if len(valid_groups) < 3:
                st.error("⚠️ Для множественных сравнений нужно минимум 3 объекта, имеющих ≥3 реплики. Для парных перейдите во вкладку 'A/B Тестирование'.")
            else:
                st.markdown("#### 1. Оценка допущений (Assumptions)")
                c_assump1, c_assump2 = st.columns([1, 2])
                
                with c_assump1:
                    shapiro_p = min([stats.shapiro(g)[1] if np.var(g) > 0 else 1.0 for g in valid_groups])
                    levene_stat, levene_p = stats.levene(*valid_groups) if any(np.var(g) > 0 for g in valid_groups) else (0, 1.0)
                    
                    is_normal = shapiro_p > alpha_level
                    is_homoscedastic = levene_p > alpha_level
                    use_parametric = is_normal and is_homoscedastic
                    
                    st.write(f"**Тест нормальности Шапиро-Уилка:** p={shapiro_p:.4e} {'✅' if is_normal else '❌'}")
                    st.write(f"**Тест дисперсий Левена:** p={levene_p:.4e} {'✅' if is_homoscedastic else '❌'}")
                    st.info(f"💡 Алгоритм выбрал: **{'Дисперсионный анализ (ANOVA)' if use_parametric else 'Критерий Краскела — Уоллиса'}**")
                
                with c_assump2:
                    # График Квантиль-Квантиль
                    fig_qq = go.Figure()
                    for name, group in zip(group_names, groups):
                        if len(group) >= 3 and np.var(group) > 0:
                            osm, osr = stats.probplot(group, dist="norm")[0]
                            fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name=str(name)))
                    
                    fig_qq.update_layout(title="QQ-График (Визуальная оценка нормальности распределений)", 
                                         xaxis_title="Теоретические квантили (Norm)", yaxis_title="Эмпирические значения",
                                         height=300, margin=dict(t=30, b=10))
                    st.plotly_chart(fig_qq, use_container_width=True)

                st.markdown("---")
                st.markdown("#### 2. Общий тест и Апостериорный анализ (Post-Hoc)")
                if use_parametric:
                    stat, p_omnibus = stats.f_oneway(*valid_groups)
                else:
                    stat, p_omnibus = stats.kruskal(*valid_groups)
                    
                st.write(f"**p-value (Общий тест):** {p_omnibus:.4e}")
                is_significant = p_omnibus < alpha_level
                
                if is_significant:
                    st.success(f"✅ Выявлены значимые отличия. Запущен Post-Hoc анализ (Поправка Бонферрони).")
                    posthoc_df = perform_pairwise_tests(df_stat, "ID", "Значение", parametric=use_parametric)
                    
                    ph1, ph2 = st.columns([1, 1])
                    with ph1:
                        st.dataframe(posthoc_df.style.map(
                            lambda x: 'background-color: #a8e6cf; color: black' if isinstance(x, float) and x < alpha_level else '', 
                            subset=['p_adj (Bonf)']
                        ), use_container_width=True)
                        
                    with ph2:
                        matrix = pd.DataFrame(index=group_names, columns=group_names, dtype=float)
                        for _, row in posthoc_df.iterrows():
                            matrix.loc[row['Группа 1'], row['Группа 2']] = row['p_adj (Bonf)']
                            matrix.loc[row['Группа 2'], row['Группа 1']] = row['p_adj (Bonf)']
                        np.fill_diagonal(matrix.values, 1.0)
                        
                        fig_ph = px.imshow(matrix, color_continuous_scale="Reds_r", zmin=0, zmax=alpha_level,
                                           title=f"Матрица p-value (Красным подсвечены отличия p < {alpha_level})")
                        st.plotly_chart(fig_ph, use_container_width=True)
                else:
                    st.warning(f"⚠️ Статистически значимых отличий между группами в целом не выявлено (p ≥ {alpha_level}).")


        # ── ВКЛАДКА 5. A/B ТЕСТИРОВАНИЕ (ПАРАМЕТРИКА/НЕПАРАМЕТРИКА) ─────────────────
        with t5:
            st.markdown("### A/B Тестирование (Точное сравнение двух групп)")
            st.caption("Детальный анализ различий между двумя конкретными выборками (Критерий Стьюдента, Критерий Манна-Уитни, тест Шапиро-Уилка).")
            
            ab_feature = st.selectbox("1. Выберите признак (метрику) для A/B теста:", wide_df.columns, key="ab_feat")
            ab_objects = st.multiselect("2. Выберите ровно ДВА объекта (группы) для сравнения:", wide_df.index.tolist(), max_selections=2, key="ab_objs")
            
            if len(ab_objects) == 2:
                group_A_name, group_B_name = ab_objects[0], ab_objects[1]
                
                # Извлекаем сырые данные (реплики) для выбранных групп
                group_A = long_df[(long_df["ID"] == group_A_name) & (long_df["Признак"] == ab_feature)]["Значение"].dropna().values
                group_B = long_df[(long_df["ID"] == group_B_name) & (long_df["Признак"] == ab_feature)]["Значение"].dropna().values
                
                if len(group_A) >= 3 and len(group_B) >= 3:
                    st.markdown("---")
                    
                    # БЛОК МЕТРИК
                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.metric(f"Среднее: {group_A_name}", f"{np.mean(group_A):.3f}", f"n = {len(group_A)}")
                    c_m2.metric(f"Среднее: {group_B_name}", f"{np.mean(group_B):.3f}", f"n = {len(group_B)}")
                    delta = np.mean(group_B) - np.mean(group_A)
                    c_m3.metric("Разница (Delta B - A)", f"{delta:.3f}")
                    
                    st.markdown("#### 1. Оценка нормальности и дисперсий (Тесты Шапиро-Уилка и Левена)")
                    col_ab1, col_ab2 = st.columns(2)
                    
                    var_A, var_B = np.var(group_A), np.var(group_B)
                    
                    with col_ab1:
                        shapiro_A = stats.shapiro(group_A)[1] if var_A > 0 else 1.0
                        shapiro_B = stats.shapiro(group_B)[1] if var_B > 0 else 1.0
                        
                        st.write(f"**Шапиро-Уилк ({group_A_name}):** p = {shapiro_A:.4f} {'✅ Норм' if shapiro_A > alpha_level else '❌ Не норм'}")
                        st.write(f"**Шапиро-Уилк ({group_B_name}):** p = {shapiro_B:.4f} {'✅ Норм' if shapiro_B > alpha_level else '❌ Не норм'}")
                    
                    with col_ab2:
                        if var_A > 0 or var_B > 0:
                            levene_stat, levene_p = stats.levene(group_A, group_B)
                        else:
                            levene_p = 1.0
                        st.write(f"**Равенство дисперсий (Критерий Левена):** p = {levene_p:.4f} {'✅ Равны' if levene_p > alpha_level else '❌ Различны'}")
                        
                    is_parametric_ab = (shapiro_A > alpha_level) and (shapiro_B > alpha_level) and (levene_p > alpha_level)
                    
                    st.markdown("#### 2. Результаты Статистических Тестов")
                    col_t1, col_t2 = st.columns(2)
                    
                    # Выполнение тестов
                    if var_A == 0 and var_B == 0 and np.mean(group_A) == np.mean(group_B):
                        t_p, mw_p = 1.0, 1.0
                    else:
                        _, t_p = stats.ttest_ind(group_A, group_B, equal_var=(levene_p > alpha_level))
                        _, mw_p = stats.mannwhitneyu(group_A, group_B, alternative='two-sided')
                    
                    eff_size_ab = cohens_d(group_A, group_B)
                    
                    with col_t1:
                        st.info("**Параметрический критерий**")
                        st.metric("t-критерий Стьюдента (p-value)", f"{t_p:.4e}")
                        if t_p < alpha_level:
                            st.success("✅ Группы статистически различаются (по t-тесту)")
                        else:
                            st.warning("⚠️ Статистических различий нет (по t-тесту)")
                            
                    with col_t2:
                        st.info("**Непараметрический критерий**")
                        st.metric("U-критерий Манна-Уитни (p-value)", f"{mw_p:.4e}")
                        if mw_p < alpha_level:
                            st.success("✅ Группы статистически различаются (по Манну-Уитни)")
                        else:
                            st.warning("⚠️ Статистических различий нет (по Манну-Уитни)")
                            
                    st.markdown(f"**Размер эффекта (Cohen's d):** `{abs(eff_size_ab):.3f}` (Насколько физически сильна разница между выборками)")
                    
                    # Визуализация A/B
                    st.markdown("#### 3. Визуальное сравнение распределений")
                    ab_df = long_df[(long_df["Признак"] == ab_feature) & (long_df["ID"].isin([group_A_name, group_B_name]))]
                    fig_ab = px.histogram(ab_df, x="Значение", color="ID", barmode="overlay", marginal="box", 
                                          title=f"Гистограмма и Боксплот: {ab_feature}", opacity=0.7)
                    st.plotly_chart(fig_ab, use_container_width=True)

                else:
                    st.warning("⚠️ Недостаточно данных для выбранных групп (нужно минимум 3 реплики в каждой).")
            else:
                st.info("💡 Пожалуйста, выберите ровно ДВА объекта в селекторе выше для проведения A/B тестирования.")


        # ── ВКЛАДКА 6. СВОДНАЯ СТАТИСТИКА (TABLE) ───────────────────────────────
        with t6:
            st.markdown("### Подробная сводная статистика по объектам и признакам")
            st.caption("Рассчитаны ключевые метрики для всех переменных. Идеально подходит для копирования в статьи или отчеты.")
            
            stats_summary_df = long_df.groupby(['ID', 'Признак'])['Значение'].agg(
                Количество='count',
                Среднее='mean',
                Медиана='median',
                Мода=calc_mode,
                Минимум='min',
                Максимум='max',
                Ст_откл='std',
                Дисперсия='var'
            ).reset_index()
            
            numeric_cols = ['Среднее', 'Медиана', 'Мода', 'Минимум', 'Максимум', 'Ст_откл', 'Дисперсия']
            stats_summary_df[numeric_cols] = stats_summary_df[numeric_cols].round(4)
            
            st.dataframe(stats_summary_df, use_container_width=True, height=500)
            
            st.markdown("#### Экспорт таблицы")
            try:
                excel_data = convert_df_to_excel(stats_summary_df)
                st.download_button(
                    label="📥 Скачать сводную статистику в формате Excel (.xlsx)",
                    data=excel_data,
                    file_name="summary_statistics.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Модуль выгрузки Excel недоступен в данной среде. Доступна выгрузка CSV.")
                st.download_button("📥 Скачать сводную статистику в CSV", convert_df_to_csv(stats_summary_df), "summary_statistics.csv", "text/csv")


        # ── ВКЛАДКА 7. ЭКСПОРТ ДАННЫХ (VAULT) ───────────────────────────────────────────
        with t7:
            st.markdown("### Хранилище агрегированных и сырых данных")
            st.caption("Здесь вы можете скачать очищенные массивы данных после парсинга.")
            
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
    st.info("👈 Загрузите ваш набор данных (таблицу) в панели слева для начала работы.")
    st.markdown("""
    ### 🔬 Добро пожаловать в StatPack OmniLab: Data Science Edition!
    Интерактивная среда разработана для проведения полноценного статистического анализа без написания программного кода (аналог скриптов **R** и пакетов **SPSS / Statistica**).
    
    **Ключевые возможности универсальной версии:**
    * ⚖️ **A/B Тестирование (NEW):** Выделенная среда для точного попарного сравнения выборок. Автоматически выполняет **t-тесты Стьюдента**, **U-критерий Манна-Уитни**, тесты **Шапиро-Уилка** на нормальность и строит перекрывающиеся гистограммы распределений.
    * 📑 **Сводная статистика:** Автоматический расчет средних, медианы, моды, минимума/максимума, стандартного отклонения и дисперсии с выгрузкой напрямую в **Excel**.
    * 📈 **Регрессия и корреляции Пирсона:** Оценка линейных связей между признаками (расчет $R^2$ и p-value).
    * 🧠 **Автоматическая гипотеза:** Программа оценивает допущения с помощью **QQ-графиков** и сама проводит параметрические (ANOVA) или непараметрические (Kruskal-Wallis) тесты для множества групп, дополняя выводы размером эффекта (Cohen's d).
    * 🎻 **Идеальная визуализация:** Скрипичные графики, Столбчатые графики с планками погрешностей, Heatmap, Радары, PCA-Biplot и Иерархические Дендрограммы.
    """)
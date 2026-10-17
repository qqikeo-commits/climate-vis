import json
import numpy as np
import pandas as pd

print(">>> [1/6] 处理全球宏观均温、年代际阶梯与百年月度热力矩阵 (GlobalTemperatures.csv)...")
df_global = pd.read_csv('GlobalTemperatures.csv')
df_global['dt'] = pd.to_datetime(df_global['dt'])
df_global['year'] = df_global['dt'].dt.year
df_global['month'] = df_global['dt'].dt.month
df_global = df_global[df_global['year'] >= 1850].copy()

# 1. 年度宏观走势
global_annual = df_global.groupby('year').agg(
    land_temp=('LandAverageTemperature', 'mean'),
    land_uncertainty=('LandAverageTemperatureUncertainty', 'mean'),
    ocean_land_temp=('LandAndOceanAverageTemperature', 'mean'),
    max_temp=('LandMaxTemperature', 'mean'),
    min_temp=('LandMinTemperature', 'mean')
).reset_index().round(2)

# 2. 年代际汇总 (1850s - 2010s)
df_global['decade'] = (df_global['year'] // 10) * 10
decade_summary = df_global.groupby('decade').agg(
    avg_temp=('LandAverageTemperature', 'mean'),
    avg_unc=('LandAverageTemperatureUncertainty', 'mean')
).reset_index().round(2)
decade_summary['decade_str'] = decade_summary['decade'].astype(str) + 's'

# 3. 1900年以来 [年份 x 月份] 二维热力矩阵 + 四季均值
df_heatmap_src = df_global[df_global['year'] >= 1900].dropna(subset=['LandAverageTemperature']).copy()
heatmap_years = sorted(df_heatmap_src['year'].unique().tolist())
heatmap_data = []
for _, row in df_heatmap_src.iterrows():
    # [年份索引, 月份索引(0-11), 温度]
    y_idx = heatmap_years.index(int(row['year']))
    m_idx = int(row['month']) - 1
    heatmap_data.append([y_idx, m_idx, round(float(row['LandAverageTemperature']), 2)])

# 工业革命早期(1850-1900)与近现代(1990-至今)各月均温对比（用于极坐标雷达）
early_monthly = df_global[(df_global['year'] >= 1850) & (df_global['year'] <= 1900)].groupby('month')['LandAverageTemperature'].mean().round(2).tolist()
recent_monthly = df_global[df_global['year'] >= 1990].groupby('month')['LandAverageTemperature'].mean().round(2).tolist()

global_pack = {
    "annual": global_annual.to_dict(orient='records'),
    "decades": decade_summary.to_dict(orient='records'),
    "heatmap_years": [str(y) for y in heatmap_years],
    "heatmap_data": heatmap_data,
    "early_monthly": early_monthly,
    "recent_monthly": recent_monthly
}

with open('global_trend.json', 'w', encoding='utf-8') as f:
    json.dump(global_pack, f, ensure_ascii=False)
print("  -> 已生成增强版 global_trend.json (含月度热力矩阵与年代阶梯)")

# =========================================================
print(">>> [2/6] 处理各国时空演变、下钻与全球升温分布 (GlobalLandTemperaturesByCountry.csv)...")
df_country = pd.read_csv('GlobalLandTemperaturesByCountry.csv')
df_country['dt'] = pd.to_datetime(df_country['dt'])
df_country['year'] = df_country['dt'].dt.year
df_country = df_country.dropna(subset=['AverageTemperature']).copy()
df_country = df_country[df_country['year'] >= 1850]

base_country = df_country[(df_country['year'] >= 1850) & (df_country['year'] <= 1900)].groupby('Country')['AverageTemperature'].mean()
country_yearly = df_country.groupby(['Country', 'year'])['AverageTemperature'].mean().reset_index()
country_yearly['base'] = country_yearly['Country'].map(base_country)
country_yearly = country_yearly.dropna(subset=['base']).copy()
country_yearly['anomaly'] = (country_yearly['AverageTemperature'] - country_yearly['base']).round(2)
country_yearly['AverageTemperature'] = country_yearly['AverageTemperature'].round(2)

country_map_dict = {}
for yr, grp in country_yearly.groupby('year'):
    if yr % 5 == 0 or yr == country_yearly['year'].max():
        country_map_dict[int(yr)] = [{"name": row['Country'], "value": row['anomaly']} for _, row in grp.iterrows()]

with open('country_anomaly.json', 'w', encoding='utf-8') as f:
    json.dump(country_map_dict, f, ensure_ascii=False)

country_trends = {}
for country, grp in country_yearly.groupby('Country'):
    country_trends[country] = {
        "years": grp['year'].tolist(),
        "temps": grp['AverageTemperature'].tolist(),
        "anomalies": grp['anomaly'].tolist()
    }

with open('country_trends.json', 'w', encoding='utf-8') as f:
    json.dump(country_trends, f, ensure_ascii=False)
print("  -> 已生成 country_anomaly.json 与 country_trends.json")

# =========================================================
print(">>> [3/6] 处理中国各省份数据与玫瑰图分布 (GlobalLandTemperaturesByState.csv)...")
try:
    df_state = pd.read_csv('GlobalLandTemperaturesByState.csv')
    df_state['dt'] = pd.to_datetime(df_state['dt'])
    df_state['year'] = df_state['dt'].dt.year
    df_china_state = df_state[df_state['Country'] == 'China'].dropna(subset=['AverageTemperature']).copy()
    df_china_state = df_china_state[df_china_state['year'] >= 1850]

    base_state = df_china_state[(df_china_state['year'] >= 1850) & (df_china_state['year'] <= 1900)].groupby('State')['AverageTemperature'].mean()
    recent_state = df_china_state[df_china_state['year'] >= 2000].groupby('State')['AverageTemperature'].mean()

    state_diff = (recent_state - base_state).dropna().reset_index()
    state_diff.rename(columns={'AverageTemperature': 'rise'}, inplace=True)
    state_diff['rise'] = state_diff['rise'].round(2)
    state_diff = state_diff.sort_values(by='rise', ascending=False)

    with open('china_states.json', 'w', encoding='utf-8') as f:
        json.dump(state_diff.to_dict(orient='records'), f, ensure_ascii=False)
    print("  -> 已生成 china_states.json")
except Exception as e:
    print("  [提示] 省份数据跳过:", e)

# =========================================================
print(">>> [4/6] 处理主要大都市与纬度放大效应 (GlobalLandTemperaturesByMajorCity.csv)...")
df_major = pd.read_csv('GlobalLandTemperaturesByMajorCity.csv')
df_major['dt'] = pd.to_datetime(df_major['dt'])
df_major['year'] = df_major['dt'].dt.year
df_major = df_major.dropna(subset=['AverageTemperature']).copy()

def parse_lat(val):
    if isinstance(val, str):
        v = val.strip()
        num = float(v[:-1])
        return num if v.endswith('N') else -num
    return 0.0

df_major['lat_num'] = df_major['Latitude'].apply(parse_lat)
base_major = df_major[(df_major['year'] >= 1850) & (df_major['year'] <= 1900)].groupby(['Country', 'City', 'lat_num'])['AverageTemperature'].mean()
recent_major = df_major[df_major['year'] >= 2000].groupby(['Country', 'City', 'lat_num'])['AverageTemperature'].mean()

major_diff = (recent_major - base_major).dropna().reset_index()
major_diff.rename(columns={'AverageTemperature': 'temp_rise'}, inplace=True)
major_diff['temp_rise'] = major_diff['temp_rise'].round(2)

top_cities = major_diff.sort_values(by='temp_rise', ascending=False).to_dict(orient='records')
with open('top_cities.json', 'w', encoding='utf-8') as f:
    json.dump(top_cities, f, ensure_ascii=False)
print("  -> 已生成 top_cities.json")

# =========================================================
print(">>> [5/6] 提取重点国际都市百年轨迹与四季雷达画像 (已剔除北京)...")
try:
    target_cities = {
        'shanghai': 'Shanghai',
        'tokyo': 'Tokyo',
        'london': 'London',
        'new york': 'New York',
        'moscow': 'Moscow'
    }

    df_city_sample = pd.read_csv('GlobalLandTemperaturesByCity.csv', usecols=['dt', 'AverageTemperature', 'AverageTemperatureUncertainty', 'City'])
    df_city_sample['City_Clean'] = df_city_sample['City'].astype(str).str.strip().str.lower()

    df_filtered = df_city_sample[df_city_sample['City_Clean'].isin(target_cities.keys())].dropna(subset=['AverageTemperature']).copy()
    df_filtered['dt'] = pd.to_datetime(df_filtered['dt'])
    df_filtered['year'] = df_filtered['dt'].dt.year
    df_filtered['month'] = df_filtered['dt'].dt.month
    df_filtered = df_filtered[df_filtered['year'] >= 1900]

    city_lookup = {}
    for lower_name, standard_name in target_cities.items():
        sub_df = df_filtered[df_filtered['City_Clean'] == lower_name]
        if not sub_df.empty:
            annual = sub_df.groupby('year')['AverageTemperature'].mean().round(2).reset_index()
            # 提取春(3-5)、夏(6-8)、秋(9-11)、冬(12,1,2)均温与年极差用于雷达画像
            spring = sub_df[sub_df['month'].isin([3,4,5])]['AverageTemperature'].mean()
            summer = sub_df[sub_df['month'].isin([6,7,8])]['AverageTemperature'].mean()
            autumn = sub_df[sub_df['month'].isin([9,10,11])]['AverageTemperature'].mean()
            winter = sub_df[sub_df['month'].isin([12,1,2])]['AverageTemperature'].mean()
            city_lookup[standard_name] = {
                "years": annual['year'].tolist(),
                "temps": annual['AverageTemperature'].tolist(),
                "radar": [round(spring,1), round(summer,1), round(autumn,1), round(winter,1), round(summer - winter, 1)]
            }
            print(f"  -> [OK] 录入 {standard_name}: {len(annual)} 年份 + 四季雷达画像")

    with open('city_lookup.json', 'w', encoding='utf-8') as f:
        json.dump(city_lookup, f, ensure_ascii=False)
    print("  -> 已生成增强版 city_lookup.json")
except Exception as e:
    print("  [提示] 城市速查数据生成跳过:", e)

print("\n[SUCCESS] 全量高阶数据集构建完毕！")
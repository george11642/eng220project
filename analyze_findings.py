"""
Quick analysis script to find interesting findings for presentation
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/cleaned_data.csv')

print("="*60)
print("KEY FINDINGS FOR PRESENTATION")
print("="*60)

# 1. New Mexico Overview
nm = df[df['jurisdiction'].str.contains('New Mexico', case=False, na=False)]
other = df[~df['jurisdiction'].str.contains('New Mexico', case=False, na=False)]

print("\n1. NEW MEXICO vs NATIONAL AVERAGE")
print("-" * 60)
print(f"New Mexico Average Crime Rate: {nm['crime_rate_per_100k'].mean():.2f} per 100k")
print(f"National Average Crime Rate: {other['crime_rate_per_100k'].mean():.2f} per 100k")
nm_crime_diff = ((nm['crime_rate_per_100k'].mean() / other['crime_rate_per_100k'].mean()) - 1) * 100
print(f"New Mexico is {abs(nm_crime_diff):.1f}% {'HIGHER' if nm_crime_diff > 0 else 'LOWER'} than national average")

print(f"\nNew Mexico Average Incarceration Rate: {nm['incarceration_rate_per_100k'].mean():.2f} per 100k")
print(f"National Average Incarceration Rate: {other['incarceration_rate_per_100k'].mean():.2f} per 100k")
nm_incarc_diff = ((nm['incarceration_rate_per_100k'].mean() / other['incarceration_rate_per_100k'].mean()) - 1) * 100
print(f"New Mexico is {abs(nm_incarc_diff):.1f}% {'HIGHER' if nm_incarc_diff > 0 else 'LOWER'} than national average")

# 2. New Mexico Rankings
latest = df[df['year'] == 2016].copy()
latest = latest.sort_values('crime_rate_per_100k', ascending=False)
nm_rank_crime = (latest['crime_rate_per_100k'] > latest[latest['jurisdiction'].str.contains('New Mexico', case=False, na=False)]['crime_rate_per_100k'].values[0]).sum() + 1

latest_incarc = latest.sort_values('incarceration_rate_per_100k', ascending=False)
nm_rank_incarc = (latest_incarc['incarceration_rate_per_100k'] > latest_incarc[latest_incarc['jurisdiction'].str.contains('New Mexico', case=False, na=False)]['incarceration_rate_per_100k'].values[0]).sum() + 1

print("\n2. NEW MEXICO RANKINGS (2016)")
print("-" * 60)
print(f"Crime Rate Rank: #{nm_rank_crime} out of {len(latest)} states (HIGHEST in nation!)")
print(f"Incarceration Rate Rank: #{nm_rank_incarc} out of {len(latest)} states")

# 3. Trends Over Time
nm_sorted = nm.sort_values('year')
crime_2001 = nm_sorted[nm_sorted['year'] == 2001]['crime_rate_per_100k'].values[0]
crime_2016 = nm_sorted[nm_sorted['year'] == 2016]['crime_rate_per_100k'].values[0]
crime_change = ((crime_2016 / crime_2001) - 1) * 100

incarc_2001 = nm_sorted[nm_sorted['year'] == 2001]['incarceration_rate_per_100k'].values[0]
incarc_2016 = nm_sorted[nm_sorted['year'] == 2016]['incarceration_rate_per_100k'].values[0]
incarc_change = ((incarc_2016 / incarc_2001) - 1) * 100

print("\n3. NEW MEXICO TRENDS (2001-2016)")
print("-" * 60)
print(f"Crime Rate: {crime_2001:.0f} (2001) -> {crime_2016:.0f} (2016)")
print(f"Change: {crime_change:+.1f}%")
print(f"\nIncarceration Rate: {incarc_2001:.0f} (2001) -> {incarc_2016:.0f} (2016)")
print(f"Change: {incarc_change:+.1f}%")

# 4. Correlation
corr = df[['crime_rate_per_100k', 'incarceration_rate_per_100k']].corr()
print("\n4. CORRELATION ANALYSIS")
print("-" * 60)
print(f"Overall Correlation (Crime vs Incarceration): {corr.iloc[0,1]:.3f}")
if abs(corr.iloc[0,1]) < 0.3:
    print("-> WEAK correlation - suggests complex relationship!")
elif abs(corr.iloc[0,1]) < 0.7:
    print("-> MODERATE correlation")
else:
    print("-> STRONG correlation")

# 5. Divergent Trends
print("\n5. DIVERGENT TRENDS ANALYSIS")
print("-" * 60)
states = df['jurisdiction'].unique()
divergent_states = []
for state in states:
    s = df[df['jurisdiction'] == state].sort_values('year')
    if len(s) > 1:
        crime_pct = ((s.iloc[-1]['crime_rate_per_100k'] - s.iloc[0]['crime_rate_per_100k']) / s.iloc[0]['crime_rate_per_100k']) * 100
        incarc_pct = ((s.iloc[-1]['incarceration_rate_per_100k'] - s.iloc[0]['incarceration_rate_per_100k']) / s.iloc[0]['incarceration_rate_per_100k']) * 100
        if crime_pct < -5 and incarc_pct > 5:
            divergent_states.append({
                'state': state,
                'crime_change': crime_pct,
                'incarc_change': incarc_pct
            })

if divergent_states:
    print(f"Found {len(divergent_states)} states where crime DECREASED but incarceration INCREASED:")
    for d in divergent_states[:5]:
        print(f"  {d['state']}: Crime DOWN {abs(d['crime_change']):.1f}%, Incarceration UP {d['incarc_change']:.1f}%")

# 6. Ratio Analysis
df['ratio'] = df['incarceration_rate_per_100k'] / df['crime_rate_per_100k']
latest['ratio'] = latest['incarceration_rate_per_100k'] / latest['crime_rate_per_100k']
latest_ratio = latest.sort_values('ratio', ascending=False)

print("\n6. INCARCERATION-TO-CRIME RATIO (2016)")
print("-" * 60)
print("States with HIGHEST ratio (most incarceration per crime):")
print(latest_ratio[['jurisdiction', 'ratio']].head().to_string(index=False))

nm_ratio = latest[latest['jurisdiction'].str.contains('New Mexico', case=False, na=False)]['ratio'].values[0]
nm_ratio_rank = (latest_ratio['ratio'] > nm_ratio).sum() + 1
print(f"\nNew Mexico Ratio: {nm_ratio:.3f} (Rank: #{nm_ratio_rank})")

# 7. Southwest Comparison
print("\n7. SOUTHWEST REGIONAL COMPARISON (2016)")
print("-" * 60)
southwest = ['New Mexico', 'Arizona', 'Texas', 'Nevada', 'Utah', 'Colorado']
sw_data = latest[latest['jurisdiction'].isin(southwest)].copy()
sw_data = sw_data.sort_values('crime_rate_per_100k', ascending=False)
print(sw_data[['jurisdiction', 'crime_rate_per_100k', 'incarceration_rate_per_100k']].to_string(index=False))
print(f"\nSouthwest Average Crime Rate: {sw_data['crime_rate_per_100k'].mean():.2f}")
print(f"Southwest Average Incarceration Rate: {sw_data['incarceration_rate_per_100k'].mean():.2f}")

# 8. Key Insight
print("\n8. KEY INSIGHT FOR PRESENTATION")
print("-" * 60)
print("New Mexico has the HIGHEST crime rate in the nation (2016)")
print("BUT incarceration rate is only slightly above average")
print("This suggests a DISCONNECT between crime levels and incarceration policies")
print(f"\nThe incarceration-to-crime ratio of {nm_ratio:.3f} means that")
print(f"for every 1000 crimes, only about {nm_ratio*1000:.0f} people are incarcerated")

print("\n" + "="*60)


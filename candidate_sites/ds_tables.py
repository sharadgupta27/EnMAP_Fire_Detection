# coding: utf-8
"""LaTeX tables of paper/data_search.tex from the search results
(results_audit_check/data_search/ranked_tiles.csv after ds_firecover.py).

  conda.exe run -n emit python candidate_sites/ds_tables.py
writes paper/ds_tab_summary.tex, paper/ds_tab_ids.tex, paper/ds_tab_more.tex,
paper/ds_tab_appendix.tex and paper/ds_numbers.tex (macros quoted in the text).
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'results_audit_check', 'data_search')
PAPER = os.path.join(ROOT, 'paper')

# (entry, short name, datatake, tiles or None = all non-held tiles with fire, figure stem)
ENTRIES = [
    ('E1', 'Nelson House fire, N Manitoba', 'DT0000145365', None, 'ds_DT0000145365'),
    ('E2', 'Rhodes (Vati--Gennadi), Greece', 'DT0000031538', [2], 'ds_DT0000031538_2'),
    ('E3', 'Prilep area, North Macedonia', 'DT0000209826', [4], 'ds_DT0000209826_4'),
    ('E4', 'White River Fire, Washington', 'DT0000004235', [2], 'ds_DT0000004235_2'),
    ('E5', 'Bush Creek East, British Columbia', 'DT0000033805', [11], 'ds_DT0000033805_11'),
    ('E6a', 'Gifford Fire, California (1 Aug)', 'DT0000145141', [1], 'ds_DT0000145141_1'),
    ('E6b', 'Gifford Fire, California (5 Aug)', 'DT0000145873', None, 'ds_DT0000145873'),
    ('E7a', 'Puerto Patriada--Epuy\\\'en, Chubut (day)', 'DT0000174652', [2], 'ds_DT0000174652_2'),
    ('E7b', 'Puerto Patriada--Epuy\\\'en, Chubut (night)', 'DT0000174656', None, 'ds_DT0000174656'),
    ('E8', 'Ourense (Larouco), Galicia', 'DT0000149658', None, 'ds_DT0000149658'),
    ('E9', 'Uiseong complex (Cheongsong), Korea', 'DT0000120726', [6], 'ds_DT0000120726_6'),
    ('E10', 'Dry-season fires, Myanmar', 'DT0000068210', None, 'ds_DT0000068210'),
    ('E11', 'Savanna fires, Lomami, DR Congo', 'DT0000088042', [7, 12, 14, 16, 17, 18],
     'ds_DT0000088042_7-12-14-16-17-18'),
    ('E12', 'Savanna fires, Taraba, Nigeria', 'DT0000115560', [5, 8, 17], 'ds_DT0000115560_5-8-17'),
    ('E13', 'Tr\\\'opico de Cochabamba, Bolivia', 'DT0000040907', [1], 'ds_DT0000040907_1'),
]


def fmt_dt(v, cover=None):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '--'
    s = ('%+.0f' % v).replace('-', '$-$')
    if cover is not None and np.isfinite(cover):
        s += '\\,(%d)' % round(100 * cover)
    return s


def tex(s):
    return str(s).replace('_', '\\_').replace('&', '\\&').replace('%', '\\%')


def entry_rows(d, dtk, tiles):
    g = d[(d['datatake'] == dtk) & (d['n_viirs'] > 0) & ~d['held']]
    if tiles:
        g = g[g['tile_no'].astype(int).isin(tiles)]
    return g.sort_values('tile_no')


def geo_short(s):
    s = str(s)
    for k, v in (('MTG-FCI', 'MTG-FCI'), ('SEVIRI', 'SEVIRI'), ('GOES16', 'GOES-16'), ('GOES17', 'GOES-17'),
                 ('GOES18', 'GOES-18'), ('GOES19', 'GOES-19'), ('Himawari', 'Himawari$^{\\dagger}$')):
        if k in s:
            return v
    return '--'


def _slots(name):
    import pickle
    p = os.path.join(OUT, 'stage2_cache', name + '.pkl')
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else []


def geo_line(r):
    """Product, number of archived scans within +/-3 h, and the scan nearest EnMAP."""
    t = pd.Timestamp(r['acq_datetime_utc']).to_pydatetime()
    geo = str(r['geo'])
    if 'Himawari' in geo:
        return (r'Himawari-8/9 AHI covers the scene (JAXA P-Tree wildfire product, 10~min); '
                r'not read by the pipeline, availability not checked')
    if 'GOES' in geo:
        sat = geo.split()[0].lower()
        cands = glob.glob(os.path.join(OUT, 'stage2_cache', f'goes_{sat}_{t:%Y%m%d}*.pkl'))
        key = min(cands, key=lambda p: abs((pd.Timestamp(os.path.basename(p)[-16:-4]) - pd.Timestamp(t)).total_seconds())) if cands else ''
        sl = _slots(os.path.basename(key)[:-4]) if key else []
        near = min(sl, key=lambda s: abs((s - t).total_seconds())) if sl else None
        s = r'%s ABI-L2-FDCF (NOAA AWS bucket \texttt{noaa-%s}): %d full-disk scans within $\pm$3~h' % (
            sat.upper().replace('GOES', 'GOES-'), sat, len(sl))
        if near:
            s += r'; nearest scan starts %s~UTC (\path{OR_ABI-L2-FDCF-M6_G%s_s%s*})' % (
                near.strftime('%H:%M'), sat[-2:], near.strftime('%Y%j%H%M'))
        return s
    if 'MTG' in geo:
        sl = _slots(f'mtg_{t:%Y%m%d}')
        lo, hi = t - pd.Timedelta(hours=3), t + pd.Timedelta(hours=3)
        n = sum(lo <= s <= hi for s in sl)
        near = max([s for s in sl if s <= t], default=None)
        s = r'MTG-I1 FCI FRP-PIXEL (LSA SAF LSA-509): %d ten-minute files within $\pm$3~h' % n
        if near:
            s += r'; file of the EnMAP slot \path{LSA-509_MTG_MTFRPPIXEL-ListProduct_MTG-FD_%s.csv.gz}' % near.strftime('%Y%m%d%H%M')
        return s
    if 'SEVIRI' in geo:
        sl = _slots(f'sev_{t:%Y%m%d}')
        lo, hi = t - pd.Timedelta(hours=3), t + pd.Timedelta(hours=3)
        n = sum(lo <= s <= hi for s in sl)
        near = max([s for s in sl if s <= t], default=None)
        s = r'MSG SEVIRI FRP-PIXEL (LSA SAF): %d fifteen-minute slots within $\pm$3~h' % n
        if near:
            s += r'; slot of the EnMAP scan %s~UTC (\path{HDF5_LSASAF_MSG_FRP-PIXEL-ListProduct_<area>_%s})' % (
                near.strftime('%H:%M'), near.strftime('%Y%m%d%H%M'))
        return s
    return '--'


def best_pass(passes, fam):
    """nearest overpass with detections in the tile: 'platform dt'."""
    best = None
    for p in str(passes).split(';'):
        if not p or p == 'nan':
            continue
        plat, m, _k = p.split(':')
        m = float(m.replace('min', ''))
        if best is None or abs(m) < abs(best[1]):
            best = (plat, m)
    if not best:
        return '--'
    ab = {'Suomi-NPP': 'S-NPP', 'NOAA-20': 'N20', 'NOAA-21': 'N21', 'Terra': 'T', 'Aqua': 'A'}
    return r'%s\,%s' % (ab.get(best[0], best[0]), ('%+.0f' % best[1]).replace('-', '$-$'))


def summary_table(d):
    L = [r'\begin{landscape}', r'\begin{table}[p]', r'\centering',
         r'\caption{The recommended EnMAP acquisitions and the other data acquired within $\pm$3~h. '
         r'Times are minutes from the EnMAP tile (negative: before EnMAP). For ECOSTRESS, Landsat and Sentinel-2 the '
         r"number in brackets is the percentage of the entry's contemporaneous VIIRS fire detections inside the "
         r"partner's footprint (best tile). VIIRS and MODIS: nearest overpass that detected fire inside the tile "
         r'(S-NPP Suomi-NPP, N20 NOAA-20, N21 NOAA-21, T Terra, A Aqua). Geostationary: product with scans every '
         r'10~min (MTG-FCI, GOES) or 15~min (SEVIRI) through the whole window; $^{\dagger}$Himawari AHI covers the '
         r'scene but is not read by the pipeline. No EMIT scene lies within $\pm$3~h of any entry.}'
         r'\label{tab:summary}', r'\footnotesize', r'\setlength{\tabcolsep}{3.5pt}',
         r'\begin{tabular}{@{}llllrrlllllll@{}}', r'\toprule',
         r'Entry & Datatake & Tiles & EnMAP (UTC) & Cloud & VIIRS & VIIRS & MODIS & SLSTR & Geo. & ECOSTRESS '
         r'& Landsat & S2 \\',
         r' & & & & \% & det. & $\Delta t$ & $\Delta t$ & $\Delta t$ & FRP & $\Delta t$ (\%) & $\Delta t$ (\%) '
         r'& $\Delta t$ (\%) \\', r'\midrule']
    for e, name, dtk, tiles, _f in ENTRIES:
        g = entry_rows(d, dtk, tiles)
        if not len(g):
            continue
        r0 = g.loc[g['n_viirs'].idxmax()]
        t_lo, t_hi = g['acq_datetime_utc'].min(), g['acq_datetime_utc'].max()
        when = t_lo[:10] + ' ' + t_lo[11:19] + ('' if t_lo == t_hi else '--' + t_hi[11:19])
        tl = ','.join('%03d' % int(t) for t in g['tile_no'])
        if len(g) > 4:
            tl = '%03d--%03d (%d)' % (int(g['tile_no'].min()), int(g['tile_no'].max()), len(g))
        cloud = '%d' % g['cloud_pct'].max() if g['cloud_pct'].nunique() == 1 else '%d--%d' % (g['cloud_pct'].min(), g['cloud_pct'].max())
        night = '$^{\\ast}$' if bool(g['night'].any()) else ''
        sl = r0['slstr_best_dt_min']

        def part(nm):
            sub = g[g[f'{nm}_n'] > 0]
            if not len(sub):
                return '--'
            k = sub.loc[sub[f'{nm}_fire_cover'].idxmax()]
            return fmt_dt(k[f'{nm}_dt_min'], k[f'{nm}_fire_cover'])
        L.append(' & '.join([e, dtk, tl, when + night, cloud, '%d' % g['n_viirs'].sum(),
                             best_pass(';'.join(g['viirs_passes'].dropna().astype(str)), 'VIIRS'),
                             best_pass(';'.join(g['modis_passes'].dropna().astype(str)), 'MODIS'),
                             fmt_dt(sl), geo_short(r0['geo']), part('ecostress'), part('landsat'), part('s2')])
                 + r' \\')
    L += [r'\bottomrule', r'\end{tabular}',
          r'\par\smallskip\raggedright $^{\ast}$Night acquisition (about 23:30 local solar time): no reflective-band '
          r'partner (Landsat, Sentinel-2) can exist; VIIRS and MODIS night passes and the geostationary products remain.',
          r'\end{table}', r'\end{landscape}']
    return '\n'.join(L)


def ids_table(d):
    L = [r'\begingroup\scriptsize\setlength{\tabcolsep}{3pt}',
         r'\begin{longtable}{@{}p{0.9cm}p{15.6cm}@{}}',
         r'\caption{Image identifiers. EnMAP: the L2A product name of each tile in the DLR catalogue and its '
         r'acquisition time to the second (the L1B to order carries the same datatake, tile number and start time; '
         r'only the processor version and processing time in the name differ). Partners: every granule within '
         r'$\pm$3~h that covers the tile. VIIRS/MODIS: FIRMS active-fire records, listed by overpass '
         r'(platform: minutes from EnMAP: detections in the tile).}\label{tab:ids}\\',
         r'\toprule Entry & Identifiers \\ \midrule\endfirsthead',
         r'\toprule Entry & Identifiers (continued) \\ \midrule\endhead', r'\bottomrule\endfoot']
    for e, name, dtk, tiles, _f in ENTRIES:
        g = entry_rows(d, dtk, tiles)
        if not len(g):
            continue
        parts = [r'\textbf{%s}' % name]
        for _, r in g.iterrows():
            parts.append(r'EnMAP tile %03d: \path{%s} \quad %s~UTC, cloud %d\%%'
                         % (int(r['tile_no']), r['scene_id'], r['acq_datetime_utc'].replace('T', ' '),
                            r['cloud_pct']))
        r0 = g.loc[g['n_viirs'].idxmax()]
        parts.append(r'VIIRS (tile %03d): %s' % (int(r0['tile_no']), tex(str(r0['viirs_passes']).replace(';', '; '))))
        if str(r0['modis_passes']) not in ('', 'nan'):
            parts.append(r'MODIS (tile %03d): %s' % (int(r0['tile_no']), tex(str(r0['modis_passes']).replace(';', '; '))))
        if str(r0['slstr_granules']) not in ('', 'nan'):
            parts.append(r'SLSTR FRP: ' + '; '.join(r'\path{%s}' % str(x) for x in str(r0['slstr_granules']).split(';')))
        parts.append('Geostationary: ' + geo_line(r0))
        for nm, lab in (('ecostress', 'ECOSTRESS L2T LSTE'), ('landsat', 'Landsat C2 L2 ST'), ('s2', 'Sentinel-2 L2A')):
            ids = sorted({x for v in g[f'{nm}_all'].dropna() for x in str(v).split(';') if x})
            if ids:
                parts.append(lab + ': ' + '; '.join(r'\path{%s}' % str(x) for x in ids))
        L.append(e + ' & ' + r' \newline '.join(parts) + r' \\ \midrule')
    L[-1] = L[-1].replace(r' \midrule', '')
    L += [r'\end{longtable}', r'\endgroup']
    return '\n'.join(L)


def appendix_table(d):
    shown = {e[2] for e in ENTRIES}
    rows = []
    for (fk, dtk), g in d[(d['tier'].isin(['A', 'B'])) & ~d['held']].groupby(['fire_key', 'datatake']):
        if dtk in shown:
            continue
        rows.append({'dtk': dtk, 'fk': fk, 'date': g['acq_datetime_utc'].min()[:16].replace('T', ' '),
                     'tier': g['tier'].min(), 'tiles': ', '.join('%03d' % int(t) for t in sorted(g['tile_no'])),
                     'lat': g['clat'].mean(), 'lon': g['clon'].mean(), 'n': int(g['n_viirs'].sum()),
                     'cloud': g['cloud_pct'].max(), 'eco': int(g['has_eco'].any()), 'ls': int(g['has_landsat'].any()),
                     's2': int(g['has_s2'].any()), 'night': bool(g['night'].any()), 'geo': geo_short(g['geo'].iloc[0])})
    A = pd.DataFrame(rows).sort_values(['tier', 'n'], ascending=[True, False])
    region = {'iberia': 'Iberia/S France', 'emed': 'E Mediterranean', 'wus': 'W USA', 'wcan': 'W Canada',
              'ecan': 'E Canada', 'seaus': 'SE Australia', 'naus': 'N Australia', 'samz': 'Bolivia/Brazil',
              'chlarg': 'Chile/Patagonia', 'safr': 'S African savanna', 'wafr': 'Sahel/Sudanian', 'seasia': 'SE Asia'}

    def reg(fk):
        if fk.startswith('sw_'):
            return region.get(fk.split('_')[1], fk)
        return tex(fk)
    L = [r'\begingroup\scriptsize\setlength{\tabcolsep}{2.5pt}',
         r'\begin{longtable}{@{}lllp{2.6cm}lrrrcccl@{}}',
         r'\caption{All other datatakes of tier A or B (not shown in Table~\ref{tab:summary}), most VIIRS detections first. '
         r'Lat./lon.: mean of the listed tile centres. Eco/LS/S2: ECOSTRESS, Landsat, Sentinel-2 within $\pm$3~h '
         r'(tile overlap $>5$\,\%; fire coverage not checked for these rows). $^{\ast}$night.}\label{tab:appendix}\\',
         r'\toprule Datatake & Tier & Date (UTC) & Tiles & Region & Lat. & Lon. & VIIRS & Eco & LS & S2 & Geo. \\ \midrule\endfirsthead',
         r'\toprule Datatake & Tier & Date (UTC) & Tiles & Region & Lat. & Lon. & VIIRS & Eco & LS & S2 & Geo. \\ \midrule\endhead',
         r'\bottomrule\endfoot']
    yes = lambda v: r'$\bullet$' if v else ''
    for r in A.itertuples():
        L.append(' & '.join([r.dtk, r.tier, r.date + (r'$^{\ast}$' if r.night else ''), r.tiles, reg(r.fk),
                             '%.2f' % r.lat, '%.2f' % r.lon, '%d' % r.n, yes(r.eco), yes(r.ls), yes(r.s2), r.geo]) + r' \\')
    L += [r'\end{longtable}', r'\endgroup']
    return '\n'.join(L), len(A)


def main():
    d = pd.read_csv(os.path.join(OUT, 'ranked_tiles.csv'))
    if 'ecostress_fire_cover' not in d:
        raise SystemExit('run ds_firecover.py first')
    s1 = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(OUT, 'stage1*_tiles.csv'))]).drop_duplicates('scene_id')
    with open(os.path.join(PAPER, 'ds_tab_summary.tex'), 'w', encoding='utf8') as fh:
        fh.write(summary_table(d))
    with open(os.path.join(PAPER, 'ds_tab_ids.tex'), 'w', encoding='utf8') as fh:
        fh.write(ids_table(d))
    app, n_app = appendix_table(d)
    with open(os.path.join(PAPER, 'ds_tab_appendix.tex'), 'w', encoding='utf8') as fh:
        fh.write(app)
    named = pd.read_csv(os.path.join(OUT, 'stage1_tiles.csv'))
    sweep = pd.read_csv(os.path.join(OUT, 'stage1_sweep_tiles.csv'))
    nums = {
        'NtilesAll': len(s1), 'NtilesNamed': len(named), 'NtilesSweep': len(sweep),
        'NfireAll': int(((s1['n_viirs'] > 0)).sum()),
        'NstageTwo': len(d), 'NtierA': int((d['tier'] == 'A').sum()), 'NtierB': int((d['tier'] == 'B').sum()),
        'NtierC': int((d['tier'] == 'C').sum()), 'NtierD': int((d['tier'] == 'D').sum()),
        'NemitAny': int((d['emit_n'] > 0).sum()), 'NecoAny': int((d['ecostress_n'] > 0).sum()),
        'NdtkStageTwo': d['datatake'].nunique(), 'Nappendix': n_app,
        'NnamedFires': len(__import__('ds_fires').all_fires()),
        'NnamedFiresWithTiles': named['fire_key'].nunique(),
    }
    with open(os.path.join(PAPER, 'ds_numbers.tex'), 'w', encoding='utf8') as fh:
        for k, v in nums.items():
            fh.write('\\newcommand{\\%s}{%s}\n' % (k, f'{v:,}'.replace(',', '{,}')))
    print(nums)


if __name__ == '__main__':
    main()

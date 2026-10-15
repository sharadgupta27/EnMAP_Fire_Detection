#!/usr/bin/env python
# coding: utf-8
"""
Keys and logins for the partner archives: entered once (the GUI's Accounts
window), kept in the standard files, read from there on every later run.

    service    fields               stored in                      used for
    earthdata  username, password   netrc: urs.earthdata.nasa.gov  ECOSTRESS, VIIRS LST, HLS (co-registration)
    cdse       username, password   netrc: identity.dataspace...   Sentinel-3 SLSTR LST and FRP
    lsasaf     username, password   netrc: datalsasaf.lsasvcs...   LSA SAF SEVIRI FRP-PIXEL
    firms      map_key              .env:  FIRMS_MAP_KEY           NASA FIRMS active fires

netrc is the file earthaccess itself reads: $NETRC if set, else ~/_netrc on
Windows and ~/.netrc elsewhere (Fire_analysis_with_EnMAP._earthdata_netrc_path
uses the same rule), so one file serves every login. The FIRMS key is not a
login, so it goes to ~/.env as FIRMS_MAP_KEY=... ($ENMAP_FIRE_ENV_FILE
overrides the path).

Lookup order everywhere: an explicit argument, then the environment variable
(EARTHDATA_USERNAME/_PASSWORD, CDSE_USERNAME/_PASSWORD,
LSASAF_USERNAME/_PASSWORD, FIRMS_MAP_KEY), then the file.

Both files are plain text, as netrc always is; they live in the user's home
folder (private to the account on Windows) and are created owner-read/write
only (0600) elsewhere.
"""
import os
import re

SERVICES = {
    'earthdata': {'kind': 'netrc', 'host': 'urs.earthdata.nasa.gov',
                  'env': {'username': 'EARTHDATA_USERNAME', 'password': 'EARTHDATA_PASSWORD'},
                  'title': 'NASA Earthdata', 'signup': 'https://urs.earthdata.nasa.gov/users/new',
                  'used_for': 'ECOSTRESS and VIIRS LST downloads, HLS reference for co-registration'},
    'cdse': {'kind': 'netrc', 'host': 'identity.dataspace.copernicus.eu',
             'env': {'username': 'CDSE_USERNAME', 'password': 'CDSE_PASSWORD'},
             'title': 'Copernicus Data Space', 'signup': 'https://dataspace.copernicus.eu',
             'used_for': 'Sentinel-3 SLSTR LST and FRP downloads'},
    'lsasaf': {'kind': 'netrc', 'host': 'datalsasaf.lsasvcs.ipma.pt',
               'env': {'username': 'LSASAF_USERNAME', 'password': 'LSASAF_PASSWORD'},
               'title': 'LSA SAF', 'signup': 'https://lsa-saf.eumetsat.int',
               'used_for': 'SEVIRI FRP-PIXEL downloads (Europe and Africa)'},
    'firms': {'kind': 'env', 'env': {'map_key': 'FIRMS_MAP_KEY'},
              'title': 'NASA FIRMS', 'signup': 'https://firms.modaps.eosdis.nasa.gov/api/map_key',
              'used_for': 'VIIRS 375 m / MODIS active-fire detections'},
}
ENV = {s: d['env'] for s, d in SERVICES.items()}
SIGNUP = {s: d['signup'] for s, d in SERVICES.items()}


def netrc_path():
    if os.environ.get('NETRC'):
        return os.environ['NETRC']
    return os.path.join(os.path.expanduser('~'), '_netrc' if os.name == 'nt' else '.netrc')


def env_path():
    return os.environ.get('ENMAP_FIRE_ENV_FILE') or os.path.join(os.path.expanduser('~'), '.env')


def location(service):
    return netrc_path() if SERVICES[service]['kind'] == 'netrc' else env_path()


# ----------------------------------------------------------------- files
def _read(path):
    if not os.path.exists(path):
        return ''
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _write(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path + '.tmp', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
        fh.write(text)
    os.replace(path + '.tmp', path)
    if os.name != 'nt':
        os.chmod(path, 0o600)


def _netrc_entry_re(host):
    # one entry: from 'machine <host>' up to the next machine/default or the end;
    # works for the one-line and the multi-line layouts
    return re.compile(r'(?s)\bmachine\s+' + re.escape(host) + r'\s.*?(?=\bmachine\s|\bdefault\b|\Z)')


def _netrc_get(host):
    m = _netrc_entry_re(host).search(_read(netrc_path()))
    if not m:
        return {}
    tok = m.group(0).split()
    out = {}
    for key, field in (('login', 'username'), ('password', 'password')):
        if key in tok and tok.index(key) + 1 < len(tok):
            out[field] = tok[tok.index(key) + 1]
    return out


def _netrc_set(host, username=None, password=None):
    text = _netrc_entry_re(host).sub('', _read(netrc_path())).strip()
    if username and password:
        text = (text + '\n\n' if text else '') + f'machine {host}\nlogin {username}\npassword {password}'
    _write(netrc_path(), text + '\n' if text else '')


def _env_get(name):
    for line in _read(env_path()).splitlines():
        m = re.match(r'\s*(?:export\s+)?' + re.escape(name) + r'\s*=\s*(.*?)\s*$', line)
        if m:
            return m.group(1).strip('\'"') or None
    return None


def _env_set(name, value):
    lines = [l for l in _read(env_path()).splitlines()
             if not re.match(r'\s*(?:export\s+)?' + re.escape(name) + r'\s*=', l)]
    if value:
        lines.append(f'{name}={value}')
    _write(env_path(), '\n'.join(lines) + '\n' if lines else '')


# ----------------------------------------------------------------- API
def stored(service):
    """What the file holds for `service` ({} when nothing)."""
    s = SERVICES[service]
    if s['kind'] == 'netrc':
        return _netrc_get(s['host'])
    return {f: v for f, env in s['env'].items() if (v := _env_get(env))}


def save(service, **fields):
    s = SERVICES[service]
    if s['kind'] == 'netrc':
        _netrc_set(s['host'], fields.get('username'), fields.get('password'))
    else:
        for f, env in s['env'].items():
            _env_set(env, fields.get(f))


def forget(service):
    save(service)                                 # no fields: the entry is removed


def get(service, **given):
    """Every field of `service`: explicit value > environment > file (None where missing)."""
    out, kept = {}, None
    for field, env in SERVICES[service]['env'].items():
        v = given.get(field) or os.environ.get(env)
        if not v:
            if kept is None:
                kept = stored(service)
            v = kept.get(field)
        out[field] = v or None
    return out


def available(service):
    return all(get(service).values())


def source(service):
    """'environment', 'file' or None - where the values come from (for display)."""
    if all(os.environ.get(e) for e in SERVICES[service]['env'].values()):
        return 'environment'
    return 'file' if available(service) else None


if __name__ == '__main__':                       # what is set up (never the values)
    for s in SERVICES:
        print(f'{s:10s} {source(s) or "-":12s} {location(s)}')

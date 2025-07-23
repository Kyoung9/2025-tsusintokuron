# constants.py

# NSL-KDDデータセットの列名（全42列 + ラベル）
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

# カテゴリ変数の列名
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

# カテゴリ変数の取りうる値（train/testデータ全体に基づく）
CATEGORICAL_VALUES = {
    "protocol_type": ["icmp", "tcp", "udp"],
    "service": [
        "aol", "auth", "bgp", "courier", "csnet_ns", "ctf", "daytime", "discard", "domain",
        "domain_u", "echo", "eco_i", "ecr_i", "efs", "exec", "finger", "ftp", "ftp_data",
        "gopher", "harvest", "hostnames", "http", "http_2784", "http_443", "http_8001",
        "imap4", "IRC", "iso_tsap", "klogin", "kshell", "ldap", "link", "login", "mtp",
        "name", "netbios_dgm", "netbios_ns", "netbios_ssn", "netstat", "nnsp", "nntp",
        "ntp_u", "other", "pm_dump", "pop_2", "pop_3", "printer", "private", "red_i",
        "remote_job", "rje", "shell", "smtp", "sql_net", "ssh", "sunrpc", "supdup",
        "systat", "telnet", "tftp_u", "tim_i", "time", "urh_i", "urp_i", "uucp",
        "uucp_path", "vmnet", "whois", "X11", "Z39_50"
    ],
    "flag": [
        "OTH", "REJ", "RSTO", "RSTOS0", "RSTR", "S0", "S1", "S2", "S3", "SF", "SH"
    ]
}

# 全41種類のクラス（ラベル）をグループ化するためのベースクラス
CLASS_NAMES_FULL = [
    'normal',
    'back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land',
    'loadmodule', 'multihop', 'neptune', 'nmap', 'perl', 'phf', 'pod', 'portsweep',
    'rootkit', 'satan', 'smurf', 'spy', 'teardrop', 'warezclient', 'warezmaster',
    'apache2', 'worm', 'xlock', 'xsnoop', 'mscan', 'saint', 'sendmail', 'named',
    'snmpgetattack', 'snmpguess', 'httptunnel', 'processtable', 'ps', 'sqlattack',
    'udpstorm', 'xterm', 'mailbomb'
]

# 4分類用のクラスラベル
CLASS_NAMES_4 = ['normal', 'DoS', 'Probe', 'R2L', 'U2R']

# 4クラス分類へのマッピング（KDDの定義に基づく）
CLASS_MAP_4_NAME = {
    'normal': 'normal',
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
    'apache2': 'DoS', 'udpstorm': 'DoS', 'processtable': 'DoS', 'mailbomb': 'DoS', 'worm': 'DoS',
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L', 'multihop': 'R2L',
    'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L',
    'snmpguess': 'R2L', 'snmpgetattack': 'R2L', 'httptunnel': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'perl': 'U2R',
    'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

# 4クラス分類に対する整数ラベルマップ
CLASS_MAP_4 = {
    name: i for i, name in enumerate(CLASS_NAMES_4)
}

# binary: normal vs attack
CLASS_MAP_BINARY = {
    'normal': 0,
    **{k: 1 for k in CLASS_NAMES_FULL if k != 'normal'}
}

# full: 各攻撃ラベルに一意な整数を割り当て
CLASS_MAP_FULL = {
    name: i for i, name in enumerate(CLASS_NAMES_FULL)
}

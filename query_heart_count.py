import sqlite3

def main():
    conn = sqlite3.connect('dbs/heart_disease.db')
    cur = conn.cursor()
    cur.execute('PRAGMA table_info(heart_disease)')
    cols = [r[1] for r in cur.fetchall()]
    print('COLUMNS:', cols)

    tried = []
    total = None

    candidates = [
        ("sex","cp",1,3),
        ("sex","chest_pain_type",1,3),
        ("gender","cp","male",3),
        ("sex_male","cp",1,3)
    ]

    for sex_col, cp_col, sex_val, cp_val in candidates:
        if sex_col in cols and cp_col in cols:
            q = f"SELECT COUNT(*) FROM heart_disease WHERE {sex_col}=? AND {cp_col}=?"
            try:
                cur.execute(q, (sex_val, cp_val))
                total = cur.fetchone()[0]
                print(f"COUNT using {sex_col}={sex_val} and {cp_col}={cp_val}: {total}")
                break
            except Exception as e:
                tried.append((sex_col, cp_col, str(e)))

    if total is None:
        print('Could not match expected column names. Tried:', tried)

    conn.close()

if __name__ == '__main__':
    main()



import sqlite3
import numpy as np

from enum import Enum


class RowEnum(str, Enum):
    X = 'x'
    Y = 'y'
    Z = 'z'
    PORO = 'porosity'
    PERM = 'permeability'
    LABEL = "labels"
    C_ID = "cube_id"
    GROUP = "RTX"
    KIND = 'kind'


class SqliteDataset:

    def __init__(self, db_path) -> None:
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        """Ouvre la connexion à la base de données lors de l'entrée dans le bloc 'with'."""
        self.conn = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ferme la connexion à la base de données lors de la sortie du bloc 'with'."""
        if self.conn:
            self.conn.close()

    def liste_cubes(self):
        query = "SELECT DISTINCT(cube_id) FROM dataset;"
        cursor = self.conn.cursor()
        cursor.execute(query)
        return [table[0] for table in cursor.fetchall()]

    def get_subcubes(self, kind='train', volumes=[4419], fields=[RowEnum.X, RowEnum.Y, RowEnum.Z, RowEnum.PERM, RowEnum.C_ID]):
        selected_fields = ", ".join([field.value for field in fields])
        placeholders = ",".join(["?"] * len(volumes))
        query = f"SELECT {selected_fields} FROM dataset WHERE cube_id IN ({placeholders}) AND kind = ?"
        cursor = self.conn.cursor()
        cursor.execute(query, volumes + [kind])
        return cursor.fetchall()

    def get_values_by_group(self, field=RowEnum.PERM):
        values_by_groups = {}
        query = f"SELECT RTX, GROUP_CONCAT({field}, ', ') AS value FROM dataset GROUP BY RTX;"
        print(query)
        cursor = self.conn.cursor()
        cursor.execute(query)
        for table in cursor.fetchall():
            values = table[1].split(',')
            values_by_groups[table[0]] = np.array(values, dtype=float)
        return values_by_groups


if __name__ == "__main__":
    db_path = "/work/lecomtje/Repositories/mixsim3d_gretsi/tests/dataset_random.db"
    with SqliteDataset(db_path=db_path) as db:
        # cubes = db.liste_cubes()
        # print(cubes)
        sub_cubes = db.get_subcubes(volumes=[4419])
        for s in sub_cubes:
            print(s)

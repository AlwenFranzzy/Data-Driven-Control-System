# environment.py
# -------------------------------------------------------------------
# Modul ini mendefinisikan Environment simulasi sistem daur ulang air.
# Environment mengatur:
# - Stok air daur ulang
# - Pemakaian air PDAM dan konversinya menjadi air daur ulang
# - Evaporasi, overflow, serta mekanisme waktu (jam & hari)
# - Perhitungan reward
# -------------------------------------------------------------------

import random

# ===================================================================
# SENSOR MODULE
# ===================================================================
class SensorModule:
    """
    Menghasilkan nilai demand (kebutuhan air) berdasarkan pola harian:
    - Pagi & sore: demand tinggi
    - Siang hari: demand menengah
    - Malam: demand rendah
    Terdapat sedikit noise acak untuk membuat simulasi lebih realistis.
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def generate_demand(self, hour, day):
        # nilai dasar demand (liter per langkah simulasi)
        base = 5.0

        # jam dengan pemakaian air lebih tinggi
        if 6 <= hour <= 9 or 17 <= hour <= 20:
            base = 12.0

        # jam siang, demand menengah
        elif 11 <= hour <= 14:
            base = 8.0

        # tambahkan noise kecil
        noise = self.rng.uniform(-1.5, 1.5)

        # demand minimal 0
        return max(0.0, base + noise)


# ===================================================================
# REWARD MODULE
# ===================================================================
class RewardModule:
    """
    Menghitung reward berdasarkan:
    - Demand terpenuhi (reward positif)
    - Penggunaan PDAM (biaya → penalti)
    - Shortage / unmet demand (penalti besar)
    - Penggunaan recycled (bonus kecil)
    Reward dikunci (clamp) agar tetap dalam -5 hingga 5.
    """

    def __init__(self, pdam_cost=2.5, shortage_penalty=2.0, recycled_bonus=1.5):
        self.pdam_cost = pdam_cost
        self.shortage_penalty = shortage_penalty
        self.recycled_bonus = recycled_bonus

    def compute(self, demand, pdam_used, recycled_used):
        reward = 0.0
        total_supplied = pdam_used + recycled_used

        # proporsi demand yang terpenuhi
        fulfilled_ratio = min(1.0, total_supplied / (demand + 1e-9))
        reward += fulfilled_ratio * 1.0

        # penalti penggunaan PDAM
        reward -= self.pdam_cost * (pdam_used / (demand + 1e-9))

        # penalti shortage apabila suplai tidak mencukupi
        if total_supplied < demand:
            shortage = demand - total_supplied
            reward -= self.shortage_penalty * (shortage / (demand + 1e-9))

        # bonus penggunaan air recycle
        reward += self.recycled_bonus * (recycled_used / (demand + 1e-9))

        # batasi reward dalam rentang [-5, 5]
        return max(-5.0, min(5.0, reward))


# ===================================================================
# ENVIRONMENT UTAMA
# ===================================================================
class WaterReuseEnv:
    """
    Environment simulasi pengelolaan air dengan 3 jenis aksi:
      0 = USE_PDAM_ONLY        → seluruh demand dipenuhi dengan PDAM
      1 = USE_RECYCLED_ONLY    → gunakan recycled; jika tidak cukup → shortage
      2 = MIX_PREFER_RECYCLED  → gunakan recycled dulu, sisanya PDAM

    Mekanisme tambahan:
    - Sebagian air PDAM yang digunakan diolah menjadi recycled pada langkah berikutnya
    - Air recycled mengalami evaporasi
    - Kapasitas tank terbatas, overflow dihitung sebagai limbah
    - Waktu bergerak per langkah: jam → hari
    """

    def __init__(
        self,
        tank_capacity=200.0,
        initial_recycled=50.0,
        treatment_rate=0.8,
        recycled_evap_rate=0.005,
        release_capacity=None,   # tidak digunakan dalam versi ini (disiapkan untuk opsi pengembangan)
        seed=None
    ):
        # parameter utama simpanan air
        self.tank_capacity = tank_capacity
        self.recycled = float(initial_recycled)

        # parameter proses
        self.treatment_rate = treatment_rate    # fraksi PDAM yang akan menjadi recycled
        self.recycled_evap_rate = recycled_evap_rate  # fraksi hilang karena evaporasi

        # modul sensor & reward
        self.sensor = SensorModule(seed=seed)
        self.reward_module = RewardModule()

        # penampung air recycle yang akan ditambahkan pada langkah berikutnya
        self.pending_recycled = 0.0

        # modul random
        self.rng = random.Random(seed)

        # waktu simulasi (dimulai dari 0)
        self.hour = 0
        self.day = 0

    # -------------------------------------------------------------------
    # RESET ENVIRONMENT
    # -------------------------------------------------------------------
    def reset(self):
        """
        Reset environment ke kondisi awal.
        State format:
        (pdam_used_prev, recycled_stock, hour, day)
        """
        self.recycled = max(0.0, min(self.tank_capacity, self.recycled))
        self.hour = 0
        self.day = 0
        self.pending_recycled = 0.0

        return (0.0, self.recycled, self.hour, self.day)

    # -------------------------------------------------------------------
    # STEP FUNCTION (INTI ENVIRONMENT)
    # -------------------------------------------------------------------
    def step(self, action):
        """
        Melakukan satu langkah simulasi dengan aksi tertentu.

        Output:
        - next_state : (pdam_used_recent, recycled_level, hour, day)
        - reward     : hasil perhitungan reward module
        - done       : selalu False (simulasi tak terbatas)
        - info       : dictionary berisi detail proses internal
        """

        # -----------------------------------------------------------
        # 1. Hitung demand berdasarkan jam & hari
        # -----------------------------------------------------------
        demand = self.sensor.generate_demand(self.hour, self.day)

        # Variabel untuk mencatat air yang digunakan
        pdam_used = 0.0
        recycled_used = 0.0

        # -----------------------------------------------------------
        # 2. Eksekusi aksi
        # -----------------------------------------------------------
        if action == 0:  # PDAM only
            pdam_used = demand

        elif action == 1:  # Recycled only
            recycled_used = min(self.recycled, demand)

        elif action == 2:  # Mixed – gunakan recycled dahulu
            recycled_used = min(self.recycled, demand)
            pdam_used = demand - recycled_used

        else:
            raise ValueError("Unknown action")

        # -----------------------------------------------------------
        # 3. Update stok recycled berdasarkan penggunaan
        # -----------------------------------------------------------
        self.recycled -= recycled_used
        self.recycled = max(0.0, self.recycled)

        # -----------------------------------------------------------
        # 4. Tambahkan hasil treatment dari PDAM (delay 1 langkah)
        # -----------------------------------------------------------
        treated = pdam_used * self.treatment_rate
        self.pending_recycled += treated

        # -----------------------------------------------------------
        # 5. Terjadi evaporasi
        # -----------------------------------------------------------
        evap_loss = self.recycled * self.recycled_evap_rate
        self.recycled -= evap_loss
        self.recycled = max(0.0, self.recycled)

        # -----------------------------------------------------------
        # 6. Tambahkan pending recycled (bisa menyebabkan overflow)
        # -----------------------------------------------------------
        self.recycled += self.pending_recycled

        if self.recycled > self.tank_capacity:
            overflow = self.recycled - self.tank_capacity
            self.recycled = self.tank_capacity
        else:
            overflow = 0.0

        # pending recycled direset setiap langkah
        self.pending_recycled = 0.0

        # -----------------------------------------------------------
        # 7. Hitung reward
        # -----------------------------------------------------------
        reward = self.reward_module.compute(
            demand, pdam_used, recycled_used
        )

        # -----------------------------------------------------------
        # 8. Update waktu
        # -----------------------------------------------------------
        self.hour = (self.hour + 1) % 24
        if self.hour == 0:
            self.day = (self.day + 1) % 7

        # -----------------------------------------------------------
        # 9. Susun next_state dan info
        # -----------------------------------------------------------
        next_state = (
            pdam_used,
            self.recycled,
            self.hour,
            self.day
        )

        info = {
            "demand": demand,
            "pdam_used": pdam_used,
            "recycled_used": recycled_used,
            "treated_added_next": treated,
            "evap_loss": evap_loss,
            "overflow": overflow
        }

        return next_state, reward, False, info

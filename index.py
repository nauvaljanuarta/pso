import random
import matplotlib.pyplot as plt
import numpy as np


def objective_function(x):
     return x**2          # f(x) = x^2

# inisialisasi 
n_particles = 10  
max_iterations = 50  
w = 0.5  
c1 = 1.5  
c2 = 1.5  
search_bounds = (-10, 10)  

particles_pos = np.random.uniform(search_bounds[0], search_bounds[1], n_particles)
particles_vel = np.random.uniform(-abs(search_bounds[1] - search_bounds[0]), abs(search_bounds[1] - search_bounds[0]), n_particles) # Kecepatan awal bisa random atau nol

pbest_pos = np.copy(particles_pos)
pbest_val = np.array([objective_function(p) for p in pbest_pos])

gbest_val = np.min(pbest_val)
gbest_pos = pbest_pos[np.argmin(pbest_val)]

best_values_per_iteration = [] 

print(f"Memulai PSO untuk fungsi f(x) = x^2")
print(f"Parameter: Partikel={n_particles}, Iterasi={max_iterations}, w={w}, c1={c1}, c2={c2}")
print(f"Batas pencarian: {search_bounds[0]} <= x <= {search_bounds[1]}\n")

for iteration in range(max_iterations):
    for i in range(n_particles):
        # Hitung nilai fitness (langkah dari slide 5)
        current_fitness = objective_function(particles_pos[i])

        # Update personal best (langkah dari slide 5)
        if current_fitness < pbest_val[i]:
            pbest_val[i] = current_fitness
            pbest_pos[i] = particles_pos[i]

        # Update global best (langkah dari slide 5)
        if current_fitness < gbest_val:
            gbest_val = current_fitness
            gbest_pos = particles_pos[i]

    # Update kecepatan dan posisi partikel
    for i in range(n_particles):
        r1 = random.random()
        r2 = random.random()

        # Update kecepatan (Persamaan 4-1 pada slide 5, meskipun tidak ditampilkan eksplisit)
        cognitive_component = c1 * r1 * (pbest_pos[i] - particles_pos[i])
        social_component = c2 * r2 * (gbest_pos - particles_pos[i])
        particles_vel[i] = w * particles_vel[i] + cognitive_component + social_component

        # Update posisi (Persamaan 4-2 pada slide 5, meskipun tidak ditampilkan eksplisit)
        particles_pos[i] = particles_pos[i] + particles_vel[i]

        # Batasi posisi partikel dalam search_bounds
        if particles_pos[i] < search_bounds[0]:
            particles_pos[i] = search_bounds[0]
            particles_vel[i] *= -0.5 # Opsi: pantulkan dan redam kecepatan
        elif particles_pos[i] > search_bounds[1]:
            particles_pos[i] = search_bounds[1]
            particles_vel[i] *= -0.5 # Opsi: pantulkan dan redam kecepatan

    best_values_per_iteration.append(gbest_val)
    if (iteration + 1) % 10 == 0 or iteration == 0:
        print(f"Iterasi {iteration + 1}/{max_iterations}: Nilai minimum sementara = {gbest_val:.6f}, Posisi x terbaik = {gbest_pos:.6f}")

# --- Cetak Hasil ---
print("\n--- Hasil Optimasi PSO ---")
print(f"Nilai minimum ditemukan: {gbest_val:.6f}") # Sesuai permintaan tugas [cite: 17]
print(f"Posisi x terbaik: {gbest_pos:.6f}") # Sesuai permintaan tugas [cite: 17]

# --- Buat Grafik ---
# Sesuai permintaan tugas [cite: 17]
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_iterations + 1), best_values_per_iteration, marker='o', linestyle='-')
plt.title('Nilai Minimum Terbaik per Iterasi (PSO)')
plt.xlabel('Iterasi')
plt.ylabel('Nilai Minimum Fungsi f(x)')
plt.grid(True)
plt.yscale('log')
plt.xticks(np.arange(0, max_iterations + 1, step=max(1, max_iterations//10))) # Membuat tick x lebih rapi
plt.tight_layout()
plt.show()
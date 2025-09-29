import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import j1
from scipy.stats import pearsonr

# Generación de Aperturas
def generar_apertura_circular(N, L, radio):
    """Genera una apertura circular en una rejilla N x N."""
    coords = np.linspace(-L / 2, L / 2, N)
    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X**2 + Y**2)
    apertura = np.zeros((N, N))
    apertura[R <= radio] = 1.0
    return apertura.astype(np.complex128)

# --- Módulo de Cálculo Teórico ---

def calcular_patron_airy(N, L_salida, longitud_onda, distancia_z, radio_apertura):
    """
    Calcula el patrón de difracción teórico de Fraunhofer (disco de Airy)
    para una apertura circular.
    """
    coords = np.linspace(-L_salida / 2, L_salida / 2, N)
    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X**2 + Y**2)
    k = 2 * np.pi / longitud_onda
    
    epsilon = 1e-9
    argumento = k * radio_apertura * R / distancia_z + epsilon
    
    intensidad = (2 * j1(argumento) / argumento)**2
    
    return intensidad / np.max(intensidad) if np.max(intensidad) > 0 else intensidad

# Núcleo - Física y Análisis de Señales 

def calcular_difraccion_fresnel_analizada(campo_entrada, L_entrada, longitud_onda, distancia_z):
    """
    Calcula la difracción de Fresnel usando el método de una sola FFT (transformada).
    """
    N, _ = campo_entrada.shape 
    k = 2 * np.pi / longitud_onda 
    delta_x0 = L_entrada / N 
    x0_coords = np.linspace(-L_entrada / 2, L_entrada / 2, N)
    X0, Y0 = np.meshgrid(x0_coords, x0_coords) 
    fase_entrada = np.exp((1j * k / (2 * distancia_z)) * (X0**2 + Y0**2)) 
    campo_preparado = campo_entrada * fase_entrada
    campo_transformado = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(campo_preparado)))
    delta_x_salida = (longitud_onda * distancia_z) / (N * delta_x0) 
    L_salida = delta_x_salida * N
    x_salida_coords = np.linspace(-L_salida / 2, L_salida / 2, N)
    X_salida, Y_salida = np.meshgrid(x_salida_coords, x_salida_coords) 
    factor_escala = (np.exp(1j * k * distancia_z) / (1j * longitud_onda * distancia_z)) * \
                      np.exp(1j * k / (2 * distancia_z) * (X_salida**2 + Y_salida**2))         
    campo_final = factor_escala * (delta_x0**2) * campo_transformado
    
    intensidad_salida = np.abs(campo_final)**2 
    return intensidad_salida, L_salida 

def verificar_muestreo_fresnel(N, L, longitud_onda, distancia_z):
    """
    Verifica si se cumple la condición de muestreo para evitar aliasing.
    """
    if longitud_onda <= 0 or N <= 0:
        return False, "Parámetros inválidos (lambda o N)."

    delta_x0 = L / N
    z_min = (N * delta_x0**2) / longitud_onda
    
    if distancia_z >= z_min:
        return True, f"Muestreo Válido (z={distancia_z:.3f}m >= z_min={z_min:.3f}m)"
    else:
        return False, f"¡Riesgo de Aliasing! (z={distancia_z:.3f}m < z_min={z_min:.3f}m)"

def espectro_angular(campo_entrada, L_entrada, longitud_onda, distancia_z):
    N, _ = campo_entrada.shape
    k=2*np.pi/longitud_onda
    delta_x0=L_entrada/N
    fx=np.fft.fftfreq(N, d=delta_x0)
    fy=np.fft.fftfreq(N, d=delta_x0)
    FoX, FoY = np.meshgrid(fx,fy, indexing='xy')
    fr2=(FoX*longitud_onda)**2+(FoY*longitud_onda)**2

    H = np.exp(1j * k * distancia_z * np.sqrt(np.maximum(0.0, 1.0 - fr2)))  # np maximum pone en 1 todo numero menor que cero
    #Se cancelan las ondas evanescentes
    H[fr2>1]=0.0

    A0 = np.fft.fft2(campo_entrada)        # Espectro angular de entrada (no centrado)
    Az = A0 * H                 # Aplicar H en el dominio de frecuencias
    Uz = np.fft.ifft2(Az)       # Campo propagado en z

    Iz = np.abs(Uz)**2
    delta_x_salida = (longitud_onda * distancia_z) / (N * delta_x0) 
    L_salida = delta_x_salida * N
    return Iz, L_salida 


if __name__ == '__main__':

    # ===================================================================
    #  PARÁMETROS DE LA SIMULACIÓN 
    # ===================================================================
    longitud_onda = 633e-9  # Longitud de onda de la luz (m)
    N = 1024                # Resolución (número de muestras en un eje, N x N)
    distancia_z = 0.157        # Distancia de propagación (m)
    L_entrada = 0.01        # Dimensión física del plano de entrada (m)
    radio = 0.17e-3      # Radio de la apertura circular (m)

    # VERIFICACIÓN DE MUESTREO 
    valido, msg = verificar_muestreo_fresnel(N, L_entrada, longitud_onda, distancia_z)
    
    print(f"Estado de la Simulación: {msg}")
    if not valido:
        print("ADVERTENCIA: Los resultados pueden no ser físicamente precisos.")

    # GENERACIÓN Y CÁLCULO 
    campo_entrada = generar_apertura_circular(N, L_entrada, radio)
    intensidad_sim, L_salida = espectro_angular(
        campo_entrada, L_entrada, longitud_onda, distancia_z)
    #intensidad_sim, L_salida = calcular_difraccion_fresnel_analizada(
        #campo_entrada, L_entrada, longitud_onda, distancia_z)
    intensidad_airy = calcular_patron_airy(
        N, L_salida, longitud_onda, distancia_z, radio)

    # CORRELACIÓN CUANTITATIVA
    sim_norm = intensidad_sim / np.max(intensidad_sim)
    airy_norm = intensidad_airy # Ya está normalizada
    corr_coef, _ = pearsonr(airy_norm.flatten(), sim_norm.flatten())
    
    print(f"Coeficiente de Correlación de Pearson: {corr_coef:.6f}")
    print("(Un valor cercano a 1.0 indica una similitud perfecta)")

    # ===================================================================
    #  VISUALIZACIÓN DE RESULTADOS 
    # ===================================================================
    
    # --- VENTANA 1: GRÁFICO DE CORRELACIÓN ---
    fig_corr = plt.figure(figsize=(5, 5))
    ax_corr = fig_corr.add_subplot(1, 1, 1)
    
    ax_corr.set_title("Correlación Píxel a Píxel")
    hb = ax_corr.hexbin(airy_norm.flatten(), sim_norm.flatten(), gridsize=50, cmap='inferno', mincnt=1)
    fig_corr.colorbar(hb, ax=ax_corr, label='Número de Píxeles')
    ax_corr.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Correlación Perfecta (y=x)')
    ax_corr.set_xlabel("Intensidad Analítica (Normalizada)")
    ax_corr.set_ylabel("Intensidad Simulación (Normalizada)")
    ax_corr.set_aspect('equal', adjustable='box')
    ax_corr.grid(True, linestyle=':')
    ax_corr.legend(loc="lower right")
    ax_corr.text(0.05, 0.95, f'Pearson r = {corr_coef:.4f}',
                 transform=ax_corr.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    fig_corr.tight_layout()

    # --- VENTANA 2: PATRONES DE INTENSIDAD ---
    fig_intensity, (ax_simulacion, ax_analitica) = plt.subplots(1, 2, figsize=(10, 5))

    radio_primer_minimo = 1.22 * longitud_onda * distancia_z / (2 * radio)
    limite_visualizacion = 3 * radio_primer_minimo

    intensidad_sim_log = np.log1p(intensidad_sim)
    intensidad_airy_log = np.log1p(intensidad_airy * np.max(intensidad_sim))
    
    # Gráfico de la Simulación
    ax_simulacion.set_title("Simulación Numérica (Espectro angular)")
    ax_simulacion.imshow(intensidad_sim_log, cmap='gray', 
                         extent=[-L_salida/2*1e3, L_salida/2*1e3, -L_salida/2*1e3, L_salida/2*1e3])
    ax_simulacion.set_xlabel("x (mm)")
    ax_simulacion.set_ylabel("y (mm)")
    ax_simulacion.set_xlim(-limite_visualizacion * 1e3, limite_visualizacion * 1e3)
    ax_simulacion.set_ylim(-limite_visualizacion * 1e3, limite_visualizacion * 1e3)
    ax_simulacion.set_aspect('equal', adjustable='box')

    # Gráfico de la Solución Analítica
    ax_analitica.set_title("Solución Analítica (Airy)")
    ax_analitica.imshow(intensidad_airy_log, cmap='gray', 
                        extent=[-L_salida/2*1e3, L_salida/2*1e3, -L_salida/2*1e3, L_salida/2*1e3])
    ax_analitica.set_xlabel("x (mm)")
    ax_analitica.set_yticklabels([])
    ax_analitica.set_xlim(-limite_visualizacion * 1e3, limite_visualizacion * 1e3)
    ax_analitica.set_ylim(-limite_visualizacion * 1e3, limite_visualizacion * 1e3)
    ax_analitica.set_aspect('equal', adjustable='box')

    fig_intensity.suptitle(f"Validación de la Difracción de Fraunhofer (z = {distancia_z} m)", fontsize=16)
    fig_intensity.tight_layout(rect=[0, 0, 1, 0.95])

    # Llamada final para mostrar todas las ventanas creadas
    plt.show()
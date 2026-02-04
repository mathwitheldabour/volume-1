import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# --- Page Setup ---
st.set_page_config(page_title="Volumes of Revolution", layout="wide")
st.markdown("""
<style>
    .main { direction: ltr; }
    h1, h2, h3 { font-family: sans-serif; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🔄 Volumes of Revolution (Disk Method)")
st.subheader("الحجوم الدورانية (طريقة الأقراص)")
st.markdown("""
**Concept:** Rotating the area under the curve $y=f(x)$ around the x-axis generates a 3D solid.
<br>
**الفكرة:** تدوير المنطقة أسفل المنحنى حول محور السينات يولد مجسماً ثلاثي الأبعاد.
""", unsafe_allow_html=True)

st.divider()

# --- Inputs ---
with st.sidebar:
    st.header("Settings / الإعدادات")
    
    # اختيار الدالة
    func_option = st.selectbox(
        "Choose Function / اختر الدالة",
        ["y = x", "y = x^2", "y = sqrt(x)", "y = sin(x) + 2"]
    )
    
    # حدود التكامل
    st.subheader("Limits / حدود التكامل")
    x_start = st.number_input("Start (a) / بداية", value=0.0, step=0.5)
    x_end = st.number_input("End (b) / نهاية", value=2.0, step=0.5)
    
    st.divider()
    
    # زاوية الدوران (للمحاكاة)
    st.info("Rotate the shape! / قم بتدوير الشكل")
    angle_deg = st.slider("Rotation Angle / زاوية الدوران", 0, 360, 360, 10)

# --- Math Logic ---
def get_func(x_vals, func_name):
    if func_name == "y = x":
        return x_vals
    elif func_name == "y = x^2":
        return x_vals**2
    elif func_name == "y = sqrt(x)":
        return np.sqrt(x_vals)
    elif func_name == "y = sin(x) + 2":
        return np.sin(x_vals) + 2
    return x_vals

# حساب الحجم الحقيقي
# V = pi * integral (f(x)^2) dx
# سنستخدم التقريب (Riemann Sum) أو التكامل المباشر للدقة
from scipy.integrate import quad

def integrand(x, func_name):
    val = 0
    if func_name == "y = x": val = x
    elif func_name == "y = x^2": val = x**2
    elif func_name == "y = sqrt(x)": val = np.sqrt(x)
    elif func_name == "y = sin(x) + 2": val = np.sin(x) + 2
    return np.pi * (val**2)

exact_vol, _ = quad(integrand, x_start, x_end, args=(func_option))

# --- Visualization ---
c1, c2 = st.columns([1, 1.5])

# 2D Plot
with c1:
    st.subheader("1. 2D Area / المنطقة المستوية")
    fig2d, ax2d = plt.subplots(figsize=(5, 4))
    
    x = np.linspace(x_start, x_end, 100)
    y = get_func(x, func_option)
    
    ax2d.plot(x, y, color='blue', linewidth=2, label=f'${func_option}$')
    ax2d.fill_between(x, y, alpha=0.3, color='blue')
    ax2d.axhline(0, color='black', linewidth=1) # x-axis
    ax2d.set_xlabel("x")
    ax2d.set_ylabel("y")
    ax2d.set_title(f"Area from {x_start} to {x_end}")
    ax2d.grid(True, alpha=0.3)
    
    # رسم شريحة (Representative Rectangle)
    mid_x = (x_start + x_end) / 2
    mid_y = get_func(np.array([mid_x]), func_option)[0]
    ax2d.add_patch(plt.Rectangle((mid_x, 0), 0.1, mid_y, color='red', alpha=0.8))
    ax2d.text(mid_x, mid_y/2, " r", color='red', fontweight='bold')
    
    st.pyplot(fig2d)
    
    # Metrics
    st.metric("Volume / الحجم", f"{exact_vol:.2f} π", delta_color="off")
    st.caption("Using Disk Method / باستخدام طريقة الأقراص")

# 3D Plot
with c2:
    st.subheader("2. 3D Solid / المجسم الدوراني")
    
    # إعداد البيانات للرسم ثلاثي الأبعاد
    fig3d = plt.figure(figsize=(6, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    # 1. Grid of x and theta
    x_3d = np.linspace(x_start, x_end, 50)
    # زاوية الدوران تعتمد على الـ Slider
    theta_max = np.radians(angle_deg)
    theta_3d = np.linspace(0, theta_max, 50)
    
    X, Theta = np.meshgrid(x_3d, theta_3d)
    
    # 2. Calculate Radius (y value)
    R = get_func(X, func_option)
    
    # 3. Convert to Cartesian coordinates (Y, Z)
    # Y corresponds to the horizontal width from axis
    # Z corresponds to vertical height
    # Rotation is around X-axis: So X stays same, Y and Z change
    Y = R * np.cos(Theta)
    Z = R * np.sin(Theta)
    
    # رسم السطح
    ax3d.plot_surface(X, Y, Z, color='#3498db', alpha=0.6, edgecolor='none')
    
    # رسم الأغطية (Caps) إذا كان الدوران كاملاً لإغلاق الشكل
    if angle_deg == 360:
        # End cap
        r_end = get_func(np.array([x_end]), func_option)[0]
        y_c = np.linspace(-r_end, r_end, 20)
        z_c = np.linspace(-r_end, r_end, 20)
        Y_c, Z_c = np.meshgrid(y_c, z_c)
        mask = Y_c**2 + Z_c**2 <= r_end**2
        ax3d.plot_surface(x_end + 0*Y_c, Y_c, Z_c, color='#2980b9', alpha=0.4) # Masking is complex in mpl, simple plane here

    # إعدادات المحاور
    ax3d.set_xlabel('X Axis')
    ax3d.set_ylabel('Y Axis')
    ax3d.set_zlabel('Z Axis')
    
    # ضبط حدود الرسم ليكون متناسقاً
    max_range = max(x_end, get_func(np.array([x_end]), func_option)[0])
    ax3d.set_xlim(0, max_range + 1)
    ax3d.set_ylim(-max_range, max_range)
    ax3d.set_zlim(-max_range, max_range)
    
    # زاوية الرؤية
    ax3d.view_init(elev=20, azim=-60)
    
    st.pyplot(fig3d)

# --- Equations ---
st.divider()
st.header("The Formula / القانون الرياضي")

st.markdown("##### Volume using Disk Method / الحجم بطريقة الأقراص:")
st.latex(r"V = \pi \int_{a}^{b} [R(x)]^2 \, dx")

st.markdown("Where $R(x)$ is the function / حيث $R$ هو نصف قطر الدوران (الدالة):")

# Dynamic Equation Display
func_latex = func_option.replace("y =", "").replace("sqrt(x)", "\sqrt{x}").replace("sin(x)", "\sin(x)")
st.latex(rf"V = \pi \int_{{{x_start}}}^{{{x_end}}} ({func_latex})^2 \, dx")

st.info("""
**Visual Note:** Notice the red strip in the 2D plot? 
When rotated, it creates one 'Disk' inside the 3D solid. Summing these disks gives the integral.
<br>
**ملاحظة بصرية:** هل تلاحظ الشريحة الحمراء في الرسم ثنائي الأبعاد؟
عند تدويرها، تشكل "قرصاً" واحداً داخل المجسم. مجموع هذه الأقراص هو ما يحسبه التكامل.
""", icon="💡")
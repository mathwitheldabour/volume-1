import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import quad

# --- Page Config ---
st.set_page_config(page_title="Advanced Calculus: Volumes", layout="wide")
st.markdown("""
<style>
    .main { direction: ltr; }
    h1, h2, h3 { font-family: sans-serif; }
    .stMetric { background-color: #f0f2f6; border-radius: 5px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("∫ Advanced Volumes Visualizer")
st.markdown("##### Washer Method (الحلقات) & Known Cross-Sections (المقاطع العرضية)")
st.divider()

# --- Helpers ---
def get_func_val(x, name):
    if name == "y = x": return x
    if name == "y = x + 1": return x + 1
    if name == "y = x^2": return x**2
    if name == "y = sqrt(x)": return np.sqrt(x)
    if name == "y = 2": return np.full_like(x, 2)
    if name == "y = 0 (x-axis)": return np.zeros_like(x)
    if name == "y = 0.5x": return 0.5 * x
    return x

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Settings / الإعدادات")
    
    # 1. Choose Method
    method = st.radio(
        "Choose Method / اختر الطريقة",
        ["Volumes of Revolution (Washers/Disks)", "Volumes by Cross-Sections"]
    )
    
    st.divider()
    
    # 2. Choose Functions
    st.subheader("Functions / الدوال")
    func_top_name = st.selectbox("Top Function (Outer Radius)", ["y = x + 1", "y = 2", "y = sqrt(x)", "y = x"], index=2)
    func_bot_name = st.selectbox("Bottom Function (Inner Radius)", ["y = x^2", "y = 0.5x", "y = 0 (x-axis)"], index=0)
    
    # 3. Limits
    st.subheader("Limits / الحدود")
    a = st.number_input("Start (a)", value=0.0, step=0.5)
    b = st.number_input("End (b)", value=1.0, step=0.5)

    # 4. Specific Controls
    angle = 360
    shape_type = "Square"
    
    if method == "Volumes of Revolution (Washers/Disks)":
        st.divider()
        angle = st.slider("Rotation Angle / زاوية الدوران", 0, 360, 270, 10)
    else:
        st.divider()
        shape_type = st.selectbox(
            "Cross-Section Shape / شكل المقطع",
            ["Square (مربع)", "Semicircle (نصف دائرة)", "Equilateral Triangle (مثلث متساوي الأضلاع)"]
        )
        num_slices = st.slider("Number of Slices / عدد الشرائح", 5, 40, 15)

# --- Calculation Logic ---

# Integration Functions
def calc_vol_washer(x):
    R = get_func_val(x, func_top_name)
    r = get_func_val(x, func_bot_name)
    # Ensure R > r for calculation, though math handles negative diff via squaring, conceptually R is outer
    return np.pi * (R**2 - r**2)

def calc_vol_cross_section(x):
    top = get_func_val(x, func_top_name)
    bot = get_func_val(x, func_bot_name)
    s = top - bot # side length or diameter
    if s < 0: s = 0 # Safety
    
    if "Square" in shape_type:
        return s**2
    elif "Semicircle" in shape_type:
        return (np.pi/8) * (s**2)
    elif "Triangle" in shape_type:
        return (np.sqrt(3)/4) * (s**2)
    return 0

# Perform Integration
if method == "Volumes of Revolution (Washers/Disks)":
    volume, _ = quad(calc_vol_washer, a, b)
    formula_latex = r"V = \pi \int_{a}^{b} ([R(x)]^2 - [r(x)]^2) \, dx"
else:
    volume, _ = quad(calc_vol_cross_section, a, b)
    if "Square" in shape_type:
        formula_latex = r"V = \int_{a}^{b} [Top - Bottom]^2 \, dx"
    elif "Semicircle" in shape_type:
        formula_latex = r"V = \frac{\pi}{8} \int_{a}^{b} [Top - Bottom]^2 \, dx"
    else:
        formula_latex = r"V = \frac{\sqrt{3}}{4} \int_{a}^{b} [Top - Bottom]^2 \, dx"

# --- Visualization ---
c1, c2 = st.columns([1, 1.5])

# LEFT: 2D Base Region
with c1:
    st.subheader("1. 2D Region / المنطقة المستوية")
    fig2d, ax2d = plt.subplots(figsize=(5, 4))
    
    x_vals = np.linspace(a, b, 100)
    y_top = get_func_val(x_vals, func_top_name)
    y_bot = get_func_val(x_vals, func_bot_name)
    
    ax2d.plot(x_vals, y_top, label="Top", color='blue')
    ax2d.plot(x_vals, y_bot, label="Bottom", color='red')
    ax2d.fill_between(x_vals, y_top, y_bot, color='purple', alpha=0.3)
    
    # Draw a representative slice
    mid_x = (a + b) / 2
    mid_top = get_func_val(mid_x, func_top_name)
    mid_bot = get_func_val(mid_x, func_bot_name)
    ax2d.plot([mid_x, mid_x], [mid_bot, mid_top], color='black', linewidth=3, linestyle='--')
    ax2d.text(mid_x, mid_top + 0.1, "slice", ha='center')

    ax2d.legend()
    ax2d.grid(True, alpha=0.3)
    st.pyplot(fig2d)
    
    # Metrics
    st.metric("Total Volume / الحجم الكلي", f"{volume:.4f}", delta_color="off")
    st.latex(formula_latex)

# RIGHT: 3D Visualization
with c2:
    st.subheader("2. 3D Model / المجسم الناتج")
    fig3d = plt.figure(figsize=(8, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    x_3d = np.linspace(a, b, 50)
    
    # --- METHOD 1: WASHERS ---
    if method == "Volumes of Revolution (Washers/Disks)":
        theta = np.linspace(0, np.radians(angle), 30)
        X, T = np.meshgrid(x_3d, theta)
        
        # Outer Surface
        R = get_func_val(X, func_top_name)
        Y_out = R * np.cos(T)
        Z_out = R * np.sin(T)
        ax3d.plot_surface(X, Y_out, Z_out, color='#3498db', alpha=0.4)
        
        # Inner Surface (The Hole)
        r_in = get_func_val(X, func_bot_name)
        Y_in = r_in * np.cos(T)
        Z_in = r_in * np.sin(T)
        # Only plot inner if it's not zero (to save performance and looks)
        if np.any(r_in > 0):
             ax3d.plot_surface(X, Y_in, Z_in, color='#e74c3c', alpha=0.6)
        
        ax3d.set_title("Rotated Solid (Outer - Blue, Inner - Red)")

    # --- METHOD 2: CROSS SECTIONS ---
    else:
        # We draw multiple polygons along the axis
        x_slices = np.linspace(a, b, num_slices)
        
        for x_k in x_slices:
            t = get_func_val(x_k, func_top_name)
            b_val = get_func_val(x_k, func_bot_name)
            s = t - b_val # size of slice
            if s <= 0: continue
            
            # Draw the 2D shape in 3D at this x coordinate
            # The base is on the XY plane (z=0 is the bottom or mid?)
            # Usually Cross sections sit ON the plane.
            # Base line: from (x_k, b_val, 0) to (x_k, t, 0) in standard math visual
            # But standard Matplotlib 3D is X, Y, Z. 
            # Let's say the base is on the XY plane, perpendicular to X-axis.
            # So the base is a line segment in Y-direction. Z is height.
            
            y_base = np.linspace(b_val, t, 2)
            z_base = np.zeros_like(y_base)
            
            verts = []
            
            if "Square" in shape_type:
                # Square sticking up in Z
                # (x, b, 0), (x, t, 0), (x, t, s), (x, b, s)
                verts = [[(x_k, b_val, 0), (x_k, t, 0), (x_k, t, s), (x_k, b_val, s)]]
                
            elif "Triangle" in shape_type:
                # Equilateral triangle
                height = (np.sqrt(3)/2) * s
                mid_y = (t + b_val) / 2
                verts = [[(x_k, b_val, 0), (x_k, t, 0), (x_k, mid_y, height)]]
                
            elif "Semicircle" in shape_type:
                # Draw a polygon arc
                theta_semi = np.linspace(0, np.pi, 15)
                # Radius = s/2. Center = (t+b)/2
                rad = s/2
                mid_y = (t + b_val) / 2
                
                # Y = center + r*cos(theta), Z = r*sin(theta)
                ys = mid_y + rad * np.cos(theta_semi) # logic check: cos goes 1 to -1. 
                # actually we need to span from b_val to t.
                # Let's parametrize simply:
                # Y goes from b_val to t. 
                ys = mid_y - rad * np.cos(theta_semi) # to map correctly
                zs = rad * np.sin(theta_semi)
                
                poly_pts = []
                for i in range(len(ys)):
                    poly_pts.append((x_k, ys[i], zs[i]))
                verts = [poly_pts]

            # Add to plot
            poly = Poly3DCollection(verts, alpha=0.6, facecolors='#2ecc71', edgecolors='white')
            ax3d.add_collection3d(poly)

        # Plot the floor functions for reference
        ax3d.plot(x_vals, y_top, np.zeros_like(x_vals), color='blue', lw=2)
        ax3d.plot(x_vals, y_bot, np.zeros_like(x_vals), color='red', lw=2)
        ax3d.set_title(f"Cross Sections: {shape_type}")
        
        # Adjust limits manually because Poly3DCollection doesn't auto-scale well
        max_height = max(get_func_val(b, func_top_name) - get_func_val(b, func_bot_name), 2)
        ax3d.set_zlim(0, max_height + 1)

    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    
    # View angle
    ax3d.view_init(elev=20, azim=-60)
    
    st.pyplot(fig3d)

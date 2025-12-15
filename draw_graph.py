import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as patches

def draw_and_save_graph():
    # 1. Setup the Canvas
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')
    
    # 2. Define Nodes: Position (x,y), Type, and Labels
    # Type: 'outcome' (Gold), 'lever' (Blue), 'context' (Gray)
    nodes = {
        # The Goal
        "Unplanned Downtime\n(Next 30d)": {"pos": (5, 3), "type": "outcome"},

        # The Actionable Levers (Difference-Makers)
        "Spare Parts\nAvailability":      {"pos": (3, 4.5), "type": "lever", "ate": "ATE: -2.1h"},
        "Maintenance\nQuality Index":     {"pos": (3, 1.5), "type": "lever", "ate": "ATE: -2.0h"},
        "Production Target\nPressure":    {"pos": (3, 3.5), "type": "lever", "ate": "ATE: +1.6h"},
        "Operator Training\nHours":       {"pos": (4, 4.5), "type": "lever", "ate": "ATE: -0.1h"},
        
        # Context / Background (The "Oxygen")
        "Repair\nDuration":               {"pos": (1, 4.5), "type": "context"},
        "Failure\nRisk":                  {"pos": (1, 3),   "type": "context"},
        "Defect Rate":                    {"pos": (2, 3.5), "type": "context"},
        "Vibration RMS":                  {"pos": (1, 1.5), "type": "context"},
        "Wear Index":                     {"pos": (2, 2.5), "type": "context"},
        "Lubrication\nInterval":          {"pos": (2, 0.5), "type": "context"},
        
        # Noise / Environmental
        "Ambient Temp":                   {"pos": (4, 0.5), "type": "noise"},
        "Humidity":                       {"pos": (5, 0.5), "type": "noise"},
        "Machine Age":                    {"pos": (2, 2.0), "type": "noise"},
    }

    # 3. Define Causal Paths (Edges)
    edges = [
        ("Repair\nDuration", "Spare Parts\nAvailability"),
        ("Spare Parts\nAvailability", "Unplanned Downtime\n(Next 30d)"),
        ("Failure\nRisk", "Defect Rate"),
        ("Defect Rate", "Production Target\nPressure"),
        ("Production Target\nPressure", "Unplanned Downtime\n(Next 30d)"),
        ("Vibration RMS", "Maintenance\nQuality Index"),
        ("Maintenance\nQuality Index", "Unplanned Downtime\n(Next 30d)"),
        ("Lubrication\nInterval", "Maintenance\nQuality Index"),
        ("Wear Index", "Failure\nRisk"),
        ("Defect Rate", "Operator Training\nHours"),
        ("Ambient Temp", "Humidity"), 
        ("Machine Age", "Wear Index"),
    ]

    # 4. Initialize Graph
    G = nx.DiGraph()
    for node, attr in nodes.items():
        G.add_node(node, **attr)
    G.add_edges_from(edges)

    # 5. Styling Configuration
    colors = {
        "outcome": {"face": "#FFD700", "edge": "#B8860B"}, # Gold
        "lever":   {"face": "#A0C4FF", "edge": "#4A90E2"}, # Blue
        "context": {"face": "#E0E0E0", "edge": "#999999"}, # Gray
        "noise":   {"face": "#F5F5F5", "edge": "#CCCCCC"}  # Light Gray
    }

    # 6. Draw Nodes (as Rounded Boxes)
    for node, attr in nodes.items():
        x, y = attr["pos"]
        style = colors[attr["type"]]
        
        # Fancy Box
        box = patches.FancyBboxPatch(
            (x - 0.4, y - 0.25), 0.8, 0.5,
            boxstyle="round,pad=0.1",
            linewidth=2 if attr["type"] in ["outcome", "lever"] else 1,
            edgecolor=style["edge"],
            facecolor=style["face"],
            zorder=2
        )
        ax.add_patch(box)
        
        # Text Label
        font_weight = 'bold' if attr["type"] in ["outcome", "lever"] else 'normal'
        ax.text(x, y, node, ha='center', va='center', fontsize=10, fontweight=font_weight, zorder=3)
        
        # ATE Label (The Red Badge)
        if "ate" in attr:
            ax.text(x, y - 0.45, attr["ate"], ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='#D32F2F', 
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=0.5),
                    zorder=4)

    # 7. Draw Edges (Curved Arrows)
    for u, v in edges:
        pos_u = nodes[u]["pos"]
        pos_v = nodes[v]["pos"]
        is_active = nodes[v]["type"] in ["outcome", "lever"]
        
        ax.annotate("",
                    xy=pos_v, xytext=pos_u,
                    arrowprops=dict(arrowstyle="->", 
                                    color="#555555" if is_active else "#BBBBBB", 
                                    lw=2 if is_active else 1,
                                    connectionstyle="arc3,rad=0.1"),
                    zorder=1)

    # 8. Clean up and Save
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Legend
    plt.text(0.2, 5.8, "■ Outcome", color="#B8860B", fontweight='bold')
    plt.text(1.2, 5.8, "■ Actionable Lever", color="#4A90E2", fontweight='bold')
    plt.text(2.7, 5.8, "■ Context", color="#999999", fontweight='bold')

    plt.title("Kairos Discovered Causal Structure (FCI Output)", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # --- SAVING THE FILE ---
    output_filename = "kairos_causal_graph.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Graph successfully saved to: {output_filename}")

if __name__ == "__main__":
    draw_and_save_graph()
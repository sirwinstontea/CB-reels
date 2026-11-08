from pathlib import Path
p = Path(__file__).resolve().parents[1] / "reelmachine" / "engine" / "scene.py"
t = p.read_text(encoding="utf-8")
old = """    def to_out(self, x, y):
        p = cv2.perspectiveTransform(np.float32([[[x, y]]]), self.H_po)[0, 0]
        return float(p[0]), float(p[1])

    def scale_at"""
new = """    def to_out(self, x, y):
        p = cv2.perspectiveTransform(np.float32([[[x, y]]]), self.H_po)[0, 0]
        return float(p[0]), float(p[1])

    def warp_mask(self, page_mask):
        '''Bool page mask -> bool output-frame mask (same warp as the ink).'''
        if not self.visible:
            return np.zeros((OUT_H, OUT_W), bool)
        by0, by1, bx0, bx1 = self.box
        f = self.f
        m = page_mask[by0:by1, bx0:bx1].astype(np.float32)
        if f > 1:
            h, w = m.shape
            if h % f == 0 and w % f == 0:
                m = m.reshape(h // f, f, w // f, f).max(axis=(1, 3))
            else:
                m = cv2.resize(m, (-(-w // f), -(-h // f)), interpolation=cv2.INTER_AREA)
        out = cv2.warpPerspective(m, self.H_co, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out > 0.35

    def scale_at"""
if old not in t:
    raise SystemExit("anchor missing in scene.py")
t = t.replace(old, new, 1)
t = t.replace(
    "only what was just inked or is still drying changes",
    "only what was just inked needs a cache refresh",
    1,
)
p.write_text(t, encoding="utf-8")
print("scene.py patched")

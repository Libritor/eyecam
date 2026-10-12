"""A flat illustration of a police officer, drawn for the eight colours: every
region is one corner of the RGB cube and several cells wide at 12 x 14, so
it survives the planes scan the way a photograph cannot. Head and shoulders:
blue cap with a yellow badge and a black visor, yellow face with black eyes
and mouth, white collar, red tie, blue uniform with a yellow badge and
buttons, on a green ground. Green is chosen so that the face/ground edge
rides on the red plane alone (the plane that has read 32 of 32 on every
sitting) and the uniform/ground edge on green and blue together.

    python targets_img/draw_officer.py          # officer_illustration.png (720 x 840)
    python targets_img/make_colour_photo.py targets_img/officer_illustration.png --name officer_ai --w 12 --h 14
"""
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 720, 840                     # 12 x 14 cells of 60 px

K, Wt = (0, 0, 0), (255, 255, 255)
R, G, B = (255, 0, 0), (0, 255, 0), (0, 0, 255)
Y = (255, 255, 0)


def draw():
    """The small features sit on the 60 px cell grid (a cell that is half one
    colour and half another has no colour of its own); the head, cap and
    shoulders keep their curves."""
    im = Image.new("RGB", (W, H), G)
    d = ImageDraw.Draw(im)
    # uniform from row 10 down: shoulders that widen to the edges
    d.polygon([(0, H), (0, 660), (150, 600), (570, 600), (720, 660), (720, H)], fill=B)
    # neck (row 9, columns 5-6) and the collar (row 9, columns 3-4 and 7-8)
    d.rectangle([300, 480, 420, 600], fill=Y)
    d.polygon([(180, 540), (300, 540), (300, 600), (240, 620), (180, 600)], fill=Wt)
    d.polygon([(420, 540), (540, 540), (540, 600), (480, 620), (420, 600)], fill=Wt)
    # tie: columns 5-6 from row 10 to the bottom, pointed
    d.polygon([(300, 600), (420, 600), (420, 780), (360, 840), (300, 780)], fill=R)
    # chest badge (columns 2-3, rows 11-12) and two buttons (column 8, rows 11 and 12)
    d.ellipse([120, 660, 240, 780], fill=Y)
    d.ellipse([480, 660, 540, 720], fill=Y)
    d.ellipse([480, 720, 540, 780], fill=Y)
    # head: rows 5-8 with the chin into row 9
    d.ellipse([180, 180, 540, 580], fill=Y)
    # ears (columns 2 and 9, rows 6-7)
    d.ellipse([130, 360, 200, 480], fill=Y)
    d.ellipse([520, 360, 590, 480], fill=Y)
    # cap: crown over rows 0-2, white band row 3, black visor row 4
    d.ellipse([150, 20, 570, 260], fill=B)
    d.rectangle([150, 120, 570, 180], fill=B)
    d.rectangle([180, 180, 540, 240], fill=Wt)
    d.polygon([(130, 240), (590, 240), (610, 300), (110, 300)], fill=K)
    d.ellipse([300, 60, 420, 180], fill=Y)              # cap badge, columns 5-6, rows 1-2
    # eyes (row 6, columns 4 and 7) and mouth (row 8, columns 5-6)
    d.ellipse([240, 360, 300, 420], fill=K)
    d.ellipse([420, 360, 480, 420], fill=K)
    d.rectangle([300, 480, 420, 540], fill=K)
    return im


def main():
    im = draw()
    out = os.path.join(HERE, "officer_illustration.png")
    im.save(out)
    print(out, im.size)


if __name__ == "__main__":
    main()

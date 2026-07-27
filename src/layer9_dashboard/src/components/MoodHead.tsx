/**
 * MoodHead.tsx — the 3D subject head.
 *
 * Renders the real human head mesh (assets/headMesh.ts — MediaPipe's
 * canonical face topology with a generated cranium and neck) with three.js.
 *
 * Two passes, which is what produces the "AR scan" look:
 *   1. a dark solid pass writing depth only, so the wireframe behind the
 *      head is hidden and you read a FRONT-facing surface, not a cage;
 *   2. an additive wireframe pass tinted by the subject's current mood.
 *
 * Interaction: drag to spin, release to resume idle rotation. All look
 * settings come from the Tweaks panel through `look`, so this component
 * re-reads them each frame rather than rebuilding the scene.
 */

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { HEAD_POSITIONS, HEAD_INDICES, HEAD_EDGES } from "../assets/headMesh";
import type { Mood } from "../types";
import { MOOD_RGB } from "../mood";

export interface HeadLook {
  /** Idle rotation speed, radians/second. */
  speed: number;
  /** Rim-glow strength. */
  glow: number;
  /** Wireframe opacity. */
  wire: number;
  /** Show the lit solid surface (off = pure wireframe over a dark occluder). */
  solid: boolean;
  /** Fixed colour name, or "mood" to follow the subject's mood. */
  colorSource: "mood" | "cyan" | "blue" | "amber" | "violet";
}

const FIXED: Record<string, [number, number, number]> = {
  cyan: [0.30, 0.85, 1.00],
  blue: [0.42, 0.62, 1.00],
  amber: [1.00, 0.72, 0.30],
  violet: [0.62, 0.45, 1.00],
};

interface Props {
  mood: Mood | null;
  look: HeadLook;
}

export default function MoodHead({ mood, look }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  // Live values the animation loop reads without re-creating the scene.
  const lookRef = useRef(look);
  const moodRef = useRef(mood);
  lookRef.current = look;
  moodRef.current = mood;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      host.innerHTML =
        '<div class="state">WebGL unavailable<div class="hint">the rest of the dashboard still works</div></div>';
      return;
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    host.appendChild(renderer.domElement);
    renderer.domElement.style.cssText =
      "position:absolute;inset:0;width:100%;height:100%;display:block;cursor:grab;touch-action:none";

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    camera.position.set(0, 0, 3.25);

    // ── geometry (shared by both passes) ─────────────────────────────────
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(HEAD_POSITIONS, 3));
    geom.setIndex(new THREE.BufferAttribute(HEAD_INDICES, 1));
    geom.computeVertexNormals();

    const wireGeom = new THREE.BufferGeometry();
    wireGeom.setAttribute("position", new THREE.BufferAttribute(HEAD_POSITIONS, 3));
    wireGeom.setIndex(new THREE.BufferAttribute(HEAD_EDGES, 1));

    const group = new THREE.Group();
    // The canonical model faces +Z but sits slightly low; nudge it up so the
    // face is centred in the panel rather than the neck.
    group.position.y = 0.06;
    scene.add(group);

    // Pass 1 — solid. Either a lit surface or a pure depth occluder.
    const solidMat = new THREE.MeshStandardMaterial({
      color: 0x223044, roughness: 0.42, metalness: 0.25,
      transparent: true, opacity: 1,
    });
    const solid = new THREE.Mesh(geom, solidMat);
    group.add(solid);

    // Pass 2 — additive wireframe on top.
    const wireMat = new THREE.LineBasicMaterial({
      color: 0x4dd8ff, transparent: true, opacity: 0.55,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const wire = new THREE.LineSegments(wireGeom, wireMat);
    group.add(wire);

    // Rim light sells the volume even when the solid pass is a dark occluder.
    const key = new THREE.DirectionalLight(0xfff0e0, 2.1);
    key.position.set(0.6, 0.8, 1.2);
    const fill = new THREE.DirectionalLight(0x6699ff, 1.1);
    fill.position.set(-0.9, 0.1, -0.4);
    scene.add(key, fill, new THREE.AmbientLight(0x404a60, 1.2));

    // ── drag to rotate ───────────────────────────────────────────────────
    let yaw = 0, pitch = 0, dragging = false, lastX = 0, lastY = 0;
    let idle = true;
    const el = renderer.domElement;
    const down = (e: PointerEvent) => {
      dragging = true; idle = false; lastX = e.clientX; lastY = e.clientY;
      el.setPointerCapture(e.pointerId); el.style.cursor = "grabbing";
    };
    const move = (e: PointerEvent) => {
      if (!dragging) return;
      yaw += (e.clientX - lastX) * 0.01;
      pitch = Math.max(-0.9, Math.min(0.9, pitch + (e.clientY - lastY) * 0.008));
      lastX = e.clientX; lastY = e.clientY;
    };
    const up = () => {
      dragging = false; el.style.cursor = "grab";
      // resume idle spin shortly after release
      setTimeout(() => { idle = true; }, 1200);
    };
    el.addEventListener("pointerdown", down);
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);

    // ── resize with the panel ────────────────────────────────────────────
    const resize = () => {
      const w = host.clientWidth || 1, h = host.clientHeight || 1;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(host);

    // ── frame loop ───────────────────────────────────────────────────────
    const colour = new THREE.Color();
    let raf = 0, prev = performance.now();

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const dt = Math.min((now - prev) / 1000, 0.1);
      prev = now;

      const L = lookRef.current;
      if (idle && !dragging) yaw += L.speed * dt;

      group.rotation.y = yaw;
      group.rotation.x = pitch;

      const rgb = L.colorSource === "mood"
        ? MOOD_RGB[moodRef.current ?? "neutral"]
        : FIXED[L.colorSource] ?? FIXED.cyan;
      colour.setRGB(rgb[0], rgb[1], rgb[2]);

      wireMat.color.copy(colour);
      wireMat.opacity = L.wire;

      if (L.solid) {
        solidMat.color.copy(colour).multiplyScalar(0.30);
        solidMat.emissive.copy(colour).multiplyScalar(0.10 * L.glow);
      } else {
        // depth-only occluder: nearly black, so only the front wires show
        solidMat.color.setRGB(0.02, 0.03, 0.05);
        solidMat.emissive.setRGB(0, 0, 0);
      }
      key.intensity = 1.4 + L.glow * 0.9;

      renderer.render(scene, camera);
    };
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      el.removeEventListener("pointerdown", down);
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", up);
      geom.dispose(); wireGeom.dispose();
      solidMat.dispose(); wireMat.dispose();
      renderer.dispose();
      host.innerHTML = "";
    };
  }, []);

  return <div ref={hostRef} style={{ position: "absolute", inset: 0 }} />;
}

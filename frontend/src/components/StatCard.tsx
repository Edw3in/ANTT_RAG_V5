import React from "react";

type StatCardProps = {
  title: string;
  value: React.ReactNode;
  subtitle?: React.ReactNode;
  right?: React.ReactNode;
};

export function StatCard({ title, value, subtitle, right }: StatCardProps) {
  return (
    <div style={styles.card}>
      <div style={styles.topRow}>
        <div style={styles.title}>{title}</div>
        {right ? <div>{right}</div> : null}
      </div>

      <div style={styles.value}>{value}</div>

      {subtitle ? <div style={styles.subtitle}>{subtitle}</div> : null}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  card: {
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 14,
    padding: 16,
    background: "rgba(255,255,255,0.03)",
    boxShadow: "0 8px 24px rgba(0,0,0,0.25)",
  },
  topRow: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 },
  title: { fontSize: 13, opacity: 0.85, letterSpacing: 0.2 },
  value: { fontSize: 22, fontWeight: 700, marginTop: 10, lineHeight: 1.15 },
  subtitle: { fontSize: 12, opacity: 0.75, marginTop: 6, lineHeight: 1.35 },
};

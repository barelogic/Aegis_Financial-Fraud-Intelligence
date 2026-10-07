/** Shared backend shapes (mirrors src/serve.py JSON). */

export type RiskBand = "critical" | "high" | "medium" | "low";

export interface FraudEvent {
  txn_id: string;
  timestamp?: string;
  account_id: string;
  amount?: number | null;
  merchant_category?: string;
  channel?: string;
  device_id?: string;
  ip_address?: string;
  city?: string;
  country?: string;
  txn_type?: string;
  dest_account_id?: string | null;
  txn_risk_score?: number;
  txn_risk_band?: string;
  txn_reasons?: string;
  txn_action?: string;
  severity?: string;
  live?: boolean;
  [key: string]: unknown;
}

export interface Health {
  ok: boolean;
  n_events: number;
  cursor: number;
  playing: boolean;
  speed: number;
  live?: boolean;
  n_live?: number;
}

export interface CaseEntity {
  id: string;
  roles?: string[];
}

export interface FraudCase {
  case_id: string;
  finding_ids?: string[];
  transaction_ids?: string[];
  entities?: CaseEntity[];
  typologies?: string[];
  evidence?: unknown[];
  severity?: string;
  severity_inputs?: Record<string, unknown>;
  severity_reason?: string;
  timeline?: Array<{ stage?: string; facts?: string }>;
}

export interface AccountDetail {
  kind: "account";
  account_id: string;
  account_risk_score?: number;
  account_risk_band?: string;
  ring_id?: string | null;
  account_reasons?: string;
  account_action?: string;
  transactions?: string[];
  n_txns?: number;
  segment?: string;
  home_city?: string;
  [key: string]: unknown;
}

export interface TransactionDetail extends FraudEvent {
  kind: "transaction";
}

export interface CaseDetail extends FraudCase {
  kind: "case";
}

export type EntityDetail = AccountDetail | TransactionDetail | CaseDetail;

export type Evaluation = Record<string, number | string | boolean | null | object | undefined>;

export type ReplayOp = "start" | "pause" | "reset";
export type ReplaySpeed = 1 | 5 | 20;

import React from "react";
import { CheckCircle, CircleX, CircleCheckBig, RotateCw, ShieldAlert } from "lucide-react";

interface ApprovalButtonsProps {
  status: string;
  inputRequest?: {
    input_type: string;
    prompt?: string;
  };
  isPlanMessage?: boolean;
  onApprove?: () => void;
  onDeny?: () => void;
  onAcceptPlan?: (text: string) => void;
  onRegeneratePlan?: () => void;
}

const ApprovalButtons: React.FC<ApprovalButtonsProps> = ({
  status,
  inputRequest,
  isPlanMessage,
  onApprove,
  onDeny,
  onAcceptPlan,
  onRegeneratePlan,
}) => {
  const [planAcceptText, setPlanAcceptText] = React.useState("");

  if (status !== "awaiting_input") {
    return null;
  }

  {/* Ali Akbar Start (Gap 2 — injection alert banner with Block/Continue buttons) */}
  if (inputRequest?.input_type === "injection_alert") {
    return (
      <div className="flex flex-col gap-2 my-2">
        {/* Alert banner */}
        <div className="rounded-md border-2 border-red-400 bg-red-50 px-3 py-2 text-red-900 text-sm">
          <div className="flex items-center gap-2 mb-1">
            <ShieldAlert className="h-5 w-5 shrink-0 text-red-600" />
            <span className="font-bold text-base">⚠ Prompt Injection Detected!</span>
          </div>
          <p className="text-xs text-red-700 mb-1">
            Suspicious content was found on this page that may try to hijack the agent.
          </p>
          {inputRequest.prompt && (
            <pre className="text-xs bg-red-100 border border-red-300 rounded p-2 whitespace-pre-wrap max-h-32 overflow-y-auto font-mono">
              {inputRequest.prompt}
            </pre>
          )}
        </div>
        {/* Action buttons — Block is primary (right), Continue is secondary (left) */}
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onApprove}
            className="bg-gray-400 hover:bg-gray-500 text-white rounded flex justify-center items-center px-2 py-1.5 transition duration-300"
          >
            <CheckCircle className="h-4 w-4 mr-1" />
            <span className="text-sm mr-1">Continue Anyway</span>
          </button>
          <button
            type="button"
            onClick={onDeny}
            className="bg-red-600 hover:bg-red-700 text-white font-bold rounded flex justify-center items-center px-3 py-1.5 transition duration-300 shadow-md"
          >
            <CircleX className="h-5 w-5 mr-1" />
            <span className="text-sm mr-1">Block Page</span>
          </button>
        </div>
      </div>
    );
  }
  {/* Ali Akbar End (Gap 2) */}

  return (
    <div className="flex gap-2 justify-start">
      {inputRequest?.input_type === "approval" ? (
        <>
          <button
            type="button"
            onClick={onApprove}
            className="bg-green-500 hover:bg-green-600 text-white rounded flex justify-center items-center px-2 py-1.5 transition duration-300"
          >
            <CheckCircle className="h-5 w-5 mr-1" />
            <span className="text-sm mr-1">Approve</span>
          </button>
          <button
            type="button"
            onClick={onDeny}
            className="bg-red-500 hover:bg-red-600 text-white rounded flex justify-center items-center px-2 py-1.5 transition duration-300"
          >
            <CircleX className="h-5 w-5 mr-1" />
            <span className="text-sm mr-1">Reject</span>
          </button>
        </>
      ) : (
        isPlanMessage && (
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => onAcceptPlan?.(planAcceptText)}
              className="bg-green-500 hover:bg-green-600 text-white rounded flex justify-center items-center px-2 py-1.5 transition duration-300"
            >
              <CircleCheckBig className="h-5 w-5 mr-1" />
              <span className="text-sm mr-1">Accept Plan</span>
            </button>
            <button
              type="button"
              onClick={onRegeneratePlan}
              className="bg-magenta-800 hover:bg-magenta-900 text-white rounded flex justify-center items-center px-2 py-1.5 transition duration-300"
            >
              <RotateCw className="h-5 w-5 mr-1" />
              <span className="text-sm mr-1">Generate New Plan</span>
            </button>
          </div>
        )
      )}
    </div>
  );
};

export default ApprovalButtons;

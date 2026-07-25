// Ali Akbar Prompt Injection Start — BUG-01 fix
// Catches render-time crashes (including the intermittent React hydration
// mismatch, console errors #418/#423) that previously left the entire app
// blank (no sidebar, no chat) with no way to recover short of a manual
// browser reload. React error boundaries can only be class components.
import * as React from "react";
import { RefreshCw } from "lucide-react";

interface Props {
  children?: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error("HALO UI crashed and was caught by ErrorBoundary:", error, info);
  }

  private handleReload = () => {
    if (typeof window !== "undefined") {
      window.location.reload();
    }
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="h-screen w-full flex items-center justify-center p-4">
          <div className="max-w-md text-center">
            <div className="text-lg font-medium text-primary mb-2">
              Something went wrong
            </div>
            <div className="text-sm text-secondary mb-4">
              HALO ran into an unexpected rendering error and stopped to avoid
              showing a blank screen. Your session and history are safe —
              reloading will restore them.
            </div>
            <button
              onClick={this.handleReload}
              className="inline-flex items-center gap-2 px-4 py-2 rounded bg-accent text-white hover:opacity-90 transition-opacity"
            >
              <RefreshCw className="w-4 h-4" />
              Reload HALO
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
// Ali Akbar Prompt Injection End

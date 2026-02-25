import { useAppStore } from '../stores/appStore';

export default function ModelLoadingBar() {
  const { modelStatus, modelLoadProgress } = useAppStore();

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm text-gray-500">
        <span>
          {modelStatus === 'loading' ? 'Loading AI model…' : 'Model load failed'}
        </span>
        {modelStatus === 'loading' && (
          <span>{modelLoadProgress}%</span>
        )}
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${
            modelStatus === 'error' ? 'bg-red-500' : 'bg-blue-500'
          }`}
          style={{ width: `${modelStatus === 'error' ? 100 : modelLoadProgress}%` }}
        />
      </div>
      {modelStatus === 'error' && (
        <p className="text-xs text-red-500">
          Failed to load model. Check your connection and reload the page.
        </p>
      )}
    </div>
  );
}

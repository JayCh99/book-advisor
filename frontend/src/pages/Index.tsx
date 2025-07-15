
import ChatInterface from "@/components/ChatInterface";

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
            AI Assistant
          </h1>
          <p className="text-gray-600">Your friendly AI companion, ready to help!</p>
        </div>
        <ChatInterface />
      </div>
    </div>
  );
};

export default Index;

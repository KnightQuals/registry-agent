from agent_engine import AgentEngine


def main():
    agent = AgentEngine()

    print("==========================================")
    print("🤖 Qwen 智能体 (Registry版) 已启动")
    print("🌍 已加载工具: 天气查询")
    print("🚪 端口配置: 8502")
    print("==========================================")

    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            if not user_input: continue
            if user_input.lower() in ["exit", "quit", "退出"]: break

            response = agent.chat(user_input)
            print(f"\n🤖 AI: {response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    main()
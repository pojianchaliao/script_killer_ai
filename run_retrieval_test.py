"""
快速测试脚本 - 一键运行检索对比测试

使用方法:
    python run_retrieval_test.py

功能:
    1. 测试三种检索策略
    2. 显示对比结果
    3. 生成评估报告
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.rag.retrieval_comparison_test import RetrievalStrategyComparator
from colorama import init, Fore, Style

init()


def main():
    """主函数"""
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🧪 三国数据检索策略对比测试{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    # 检查数据文件是否存在
    data_path = "app/data/romance_three_kingdoms.json"
    
    if not os.path.exists(data_path):
        print(f"{Fore.RED}❌ 错误：找不到数据文件 {data_path}{Style.RESET_ALL}")
        return
    
    try:
        # 创建对比器
        print(f"{Fore.YELLOW}⚙️  正在加载数据和模型...{Style.RESET_ALL}")
        comparator = RetrievalStrategyComparator(data_path)
        
        # 运行所有测试
        print(f"\n{Fore.GREEN}✅ 初始化完成，开始测试...{Style.RESET_ALL}\n")
        comparator.run_all_tests()
        
        # 完成
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✅ 测试完成！{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n{Fore.RED}❌ 用户中断测试{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ 发生错误：{e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

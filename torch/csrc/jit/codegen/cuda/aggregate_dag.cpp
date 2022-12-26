#include <torch/csrc/jit/codegen/cuda/aggregate_dag.h>
#include <torch/csrc/jit/codegen/cuda/multidevice_runtime.h>
#include <torch/csrc/jit/codegen/cuda/ir_printer.h>


namespace torch {
namespace jit {
namespace fuser {
namespace cuda {


AggregateExpr::AggregateExpr(
    IrBuilderPasskey passkey,
    Group* group)
    : Expr(passkey, ExprType::AggregateExpr), group_(group) {
}

AggregateExpr::AggregateExpr(const AggregateExpr* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner), group_(src->group_) {}

Expr* AggregateExpr::shallowCopy() const {
  auto result = IrBuilder::create<AggregateExpr>(this->group_);
  result->copyPredicatesFrom(this);
  return result;
}

bool AggregateExpr::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<AggregateExpr>()) {
    return false;
  }
  const auto other_op = other->as<AggregateExpr>();

  return group_== other_op->getGroup();
}





SendRecv::SendRecv(IrBuilderPasskey passkey, AggregateVal* out, AggregateVal* in)
    : Expr(passkey, ExprType::SendRecv), out_{out}, in_{in} {
  addOutput(out);
  addInput(in);
}

SendRecv::SendRecv(const SendRecv* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner), out_{src->out_}, in_{src->in_} {}

Expr* SendRecv::shallowCopy() const {
  auto result = IrBuilder::create<SendRecv>(this->out_, this->in_);
  result->copyPredicatesFrom(this);
  return result;
}

bool SendRecv::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<SendRecv>()) {
    return false;
  }
  // const auto other_op = other->as<SendRecv>();

  return Expr::sameAs(other);
}






AggregateVal::AggregateVal(
    IrBuilderPasskey passkey, Val* val, Group* group)
    : Val(passkey, ValType::AggregateVal, val->dtype()),
    original_val_(val), group_(group) {
      //TODO: add a mapping from original val to AggregateVal.
    }

AggregateVal::AggregateVal(const AggregateVal* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), original_val_(src->original_val_), group_(src->group_) {}


bool AggregateVal::sameAs(const Statement* other) const {
  if (this == other) {
    return true;
  }
  if (!other->isA<AggregateVal>()) {
    return false;
  }
  const auto other_aggregate_val = other->as<AggregateVal>();
  return original_val_->sameAs(other_aggregate_val->original_val_)
          && group_->unique_id == other_aggregate_val->group_->unique_id;
}





AggregateDag::AggregateDag():IrContainer(){}

void AggregateDag::build(MultiGroupFusion* fusion) {
  for (auto group_ptr: fusion->groups()) {
    auto group = group_ptr.get();
    IrContainer* container = (IrContainer*)this;
    std::vector<AggregateVal*> inputs;
    std::vector<AggregateVal*> outputs;

    for (auto output_val : group->output_vals) {
      auto val = IrBuilder::create<AggregateVal>(container, output_val, group);
      producer[output_val] = val;
      outputs.push_back(val);
    }

    for (auto input_val : group->input_vals) {
      auto val = IrBuilder::create<AggregateVal>(container, input_val, group);
      inputs.push_back(val);
      if (producer.find(input_val) != producer.end()){
        //means that input_val is not a global input and 
        // is produced by another group
        auto src = producer[input_val];
        auto sendRecv = IrBuilder::create<SendRecv>(container, val, src);
        val->setDefinition(sendRecv);
      }
    }

    auto expr = IrBuilder::create<AggregateExpr>(container, group);
    for (auto out: outputs){
      expr->addOutput(out);
      out->setDefinition(expr);
    }
    for (auto in: inputs){
      expr->addInput(in);
    }
  }
}




// std::ostream& operator<< (std::ostream &out, AggregateVal const& data) {
//     return out << "AggregateVal represents Val " <<  data.getOriginalVal()
//               << "on group " << data.getGroup() << "\n";
// }

std::ostream& operator<< (std::ostream &out, AggregateExpr const& data) {
    return out << "AggregateExpr represents Group " <<  data.getGroup() << "\n";
}

std::ostream& operator<< (std::ostream &out, SendRecv const& data) {
    out << "SendRecv with inputs ";
    for (auto input: data.inputs())
      out << input;
    out<< " and outputs ";
    for (auto output: data.outputs())
      out << output;
    out << "\n";
    return out;
}

  void AggregateDag::print(){
    IrPrinter p(std::cout);
    std::cout << "AggregateDag containing Vals {\n";
    for (auto val: vals()){
      std::cout <<"   ";
      // p.handle((AggregateVal*)val);
      std::cout <<(AggregateVal*)val;
      std::cout <<"\n";
    }
    std::cout << "}\n and Exprs {\n";
    for (auto expr: unordered_exprs()){
      std::cout <<"   " << expr <<"\n";
    }
    std::cout << "}\n" << std::endl;
  }




} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

